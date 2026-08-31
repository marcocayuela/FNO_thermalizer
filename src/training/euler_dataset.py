"""
Dataset for euler_multi_quadrants_openBC (The Well), a single physical
configuration (gamma=1.4, "Dry_air"). Ported from
RT_FNO_vs_WNO/src/training/euler_dataset.py (near-verbatim -- that
implementation is already mature and mesu-validated for exactly this file
format), adapted for thermalizer's FNOTrainingEuler/DiffusionTrainingEuler
(cf. fno_training_euler.py / diffusion_training_euler.py).

The Well's native layout for this dataset -- one HDF5 file per split/chunk,
several trajectories inside, groups `t0_fields` (scalars: density, energy,
pressure) / `t1_fields` (vectors: momentum) -- is different from
Kolmogorov/KS's own `Re<X>/train_traj/sim<N>.h5` one-trajectory-per-file
convention, and already well served by this lazy-window loader; forcing it
into the sim<N>.h5 convention would only lose the streaming behavior this
file is built around.

Stacked fields: energy, density, momentum_x, momentum_y -- 4 channels.
`pressure` is deliberately excluded: for a fixed-gamma ideal gas it is fully
determined by the other 3 conservative fields via the equation of state
(p = (gamma-1) * (energy - 0.5*(momentum_x^2+momentum_y^2)/density)) -- no
extra information, and keeping it as an output channel would just dilute the
loss over a redundant channel with no thermodynamic-consistency guarantee.
Recoverable analytically from (energy, density, momentum) if a diagnostic
ever needs it.

LAZY LOADING (mirrors the source file's own history): an eager,
load-everything version OOM-killed on mesu for this project's own use of the
same data (~34 GB just for the final concatenated train array, with
transient peaks up to ~4x that) -- this version never holds more than one
requested window in memory. Each `__getitem__` reads only the window it
needs straight off disk (h5py serves chunked storage, so a
`dataset[traj_idx, start:end]` read only touches the blocks it needs, not
the whole file); each DataLoader worker opens its own HDF5 handle on first
access (handles don't share safely across a fork).
"""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

FIELD_ORDER = ["energy", "density", "momentum_x", "momentum_y"]
N_CHANNELS = len(FIELD_ORDER)


def _peek_file_info(h5_path):
    """Reads gamma/n_traj/T/H/W without loading any field array. Note:
    this file's own `n_trajectories` file-level attribute is known to be
    wrong (e.g. -1550) -- read trajectory count from the array shape
    instead, never from that attribute."""
    with h5py.File(h5_path, "r") as f:
        gamma = float(f.attrs["gamma"])
        n_traj, T, H, W = f["t0_fields"]["density"].shape
    return gamma, n_traj, T, H, W


def compute_channel_stats_streaming(h5_paths):
    """Per-channel (mean, std), streamed one trajectory at a time -- never
    more than one trajectory (~a few hundred MB) in memory at once,
    regardless of the total number of files/trajectories."""
    count = 0
    sum_ = np.zeros(N_CHANNELS, dtype=np.float64)
    sumsq = np.zeros(N_CHANNELS, dtype=np.float64)

    for path in h5_paths:
        with h5py.File(path, "r") as f:
            n_traj = f["t0_fields"]["density"].shape[0]
            for i in range(n_traj):
                fields = [
                    f["t0_fields"]["energy"][i].astype(np.float64),
                    f["t0_fields"]["density"][i].astype(np.float64),
                    f["t1_fields"]["momentum"][i, ..., 0].astype(np.float64),
                    f["t1_fields"]["momentum"][i, ..., 1].astype(np.float64),
                ]
                for c, field in enumerate(fields):
                    sum_[c] += field.sum()
                    sumsq[c] += np.square(field).sum()
                count += fields[0].size

    mean = sum_ / count
    var = np.maximum(sumsq / count - mean ** 2, 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def load_single_trajectory(h5_path, traj_idx):
    """Loads ONE trajectory (T, H, W, 4) float32 -- used by
    evaluation/correction_eval_euler.py for the GT rollout comparison
    (loading the whole file just to keep one trajectory would be wasteful)."""
    with h5py.File(h5_path, "r") as f:
        gamma = float(f.attrs["gamma"])
        energy = f["t0_fields"]["energy"][traj_idx]
        density = f["t0_fields"]["density"][traj_idx]
        momentum = f["t1_fields"]["momentum"][traj_idx]

    T, H, W = energy.shape
    full = np.empty((T, H, W, N_CHANNELS), dtype=np.float32)
    full[..., 0] = energy
    full[..., 1] = density
    full[..., 2:4] = momentum
    return full, gamma


class LazyEulerDataset(Dataset):
    """One HDF5 file, a subset of its trajectories. Each `__getitem__` reads
    only the [start:start+k+n_rollout] window of the relevant trajectory
    straight from disk -- nothing is loaded at construction beyond shape
    metadata (T, H, W).

    Returns (x, y): x is the k input frames stacked as channels,
    (H, W, k*N_CHANNELS). y depends on prediction_mode: "state" (default) ->
    the n_rollout target frames in absolute state, (n_rollout, H, W,
    N_CHANNELS), matching what Trainer.train_epoch()'s state-mode rollout
    loop iterates over; any other value -> DELTAS between consecutive
    frames (y[i] = frame[k+i] - frame[k+i-1], the last input frame being
    frame[k-1]) -- Trainer.train_epoch()'s delta-mode branch compares the
    model's raw output directly against the target with no denormalization
    step of its own, so the target must already be a delta in the SAME
    normalized space x is in, exactly mirroring
    dataset_manager.py::SequenceDataset's own state-vs-delta convention.
    """

    def __init__(self, h5_path, traj_indices, k, n_rollout, stride, mean, std, frame_skip=1,
                 prediction_mode="state"):
        self.h5_path = h5_path
        self.traj_indices = list(traj_indices)
        self.k = k
        self.n_rollout = n_rollout
        self.stride = stride
        self.mean = mean
        self.std = std
        self.prediction_mode = prediction_mode
        # frame_skip > 1: the model is trained/predicts across frame_skip
        # native dataset frames at once (native dt=0.015s for
        # euler_multi_quadrants_openBC) instead of one -- a coarser
        # effective timestep. Single-step extrapolation error grows with
        # the size of the step, so this is the knob for making the
        # emulator's own rollout actually prone to numerical divergence
        # (negative density/energy -> NaN) within the ~100-native-frame
        # trajectory budget, rather than just drifting/blurring -- cf. the
        # frame_skip=1 baseline run, which stayed finite for all 99 steps
        # but was badly over-smoothed relative to GT.
        self.frame_skip = frame_skip
        self._file = None

        _, _, T, H, W = _peek_file_info(h5_path)
        self.T, self.H, self.W = T, H, W
        span = (k + n_rollout - 1) * frame_skip + 1
        self.n_per_traj = max(0, (T - span) // stride + 1)

    def __len__(self):
        return self.n_per_traj * len(self.traj_indices)

    def __getstate__(self):
        # h5py handles don't pickle -- force _file to None before any
        # pickling (multiprocessing "spawn", or a plain object copy) so each
        # DataLoader worker reopens its own handle on first access (see
        # _ensure_open). Without this: "TypeError: h5py objects cannot be
        # pickled" as soon as a Dataset that has already accessed an item is
        # handed to workers.
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __getitem__(self, idx):
        self._ensure_open()
        traj_pos, window_pos = divmod(idx, self.n_per_traj)
        traj_idx = self.traj_indices[traj_pos]
        start = window_pos * self.stride
        span = (self.k + self.n_rollout - 1) * self.frame_skip + 1
        end = start + span

        f = self._file
        energy = f["t0_fields"]["energy"][traj_idx, start:end:self.frame_skip]
        density = f["t0_fields"]["density"][traj_idx, start:end:self.frame_skip]
        momentum = f["t1_fields"]["momentum"][traj_idx, start:end:self.frame_skip]

        window = np.empty((self.k + self.n_rollout, self.H, self.W, N_CHANNELS), dtype=np.float32)
        window[..., 0] = energy
        window[..., 1] = density
        window[..., 2:4] = momentum
        window = (window - self.mean) / self.std

        x = window[: self.k].transpose(1, 2, 0, 3).reshape(self.H, self.W, self.k * N_CHANNELS)
        if self.prediction_mode == "state":
            y = window[self.k:]
        else:
            y = window[self.k:] - window[self.k - 1: self.k + self.n_rollout - 1]
        return torch.from_numpy(x.copy()).float(), torch.from_numpy(y.copy()).float()


class LazyEulerFirstSnapshot(Dataset):
    """Diffusion-corrector counterpart of LazyEulerDataset: one snapshot per
    `__getitem__`, no target (the diffusion model only needs an initial
    frame to noise/denoise, cf. training/dataset_manager.py::FirstSnapshot).
    Same lazy, per-worker-handle, streamed-normalization design."""

    def __init__(self, h5_path, traj_indices, stride, mean, std):
        self.h5_path = h5_path
        self.traj_indices = list(traj_indices)
        self.stride = stride
        self.mean = mean
        self.std = std
        self._file = None

        _, _, T, H, W = _peek_file_info(h5_path)
        self.T, self.H, self.W = T, H, W
        self.n_per_traj = max(0, (T - 1) // stride + 1)

    def __len__(self):
        return self.n_per_traj * len(self.traj_indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __getitem__(self, idx):
        self._ensure_open()
        traj_pos, window_pos = divmod(idx, self.n_per_traj)
        traj_idx = self.traj_indices[traj_pos]
        t = window_pos * self.stride

        f = self._file
        frame = np.empty((self.H, self.W, N_CHANNELS), dtype=np.float32)
        frame[..., 0] = f["t0_fields"]["energy"][traj_idx, t]
        frame[..., 1] = f["t0_fields"]["density"][traj_idx, t]
        frame[..., 2:4] = f["t1_fields"]["momentum"][traj_idx, t]
        frame = (frame - self.mean) / self.std
        return torch.from_numpy(frame.copy()).float()


def _build_lazy_loader(dataset_cls, h5_paths_and_indices, extra_args, mean, std,
                       batch_size, num_workers, shuffle, extra_kwargs=None):
    # extra_kwargs (not just more positional extra_args): LazyEulerDataset's
    # constructor has frame_skip AFTER mean/std (frame_skip=1) -- passing it
    # positionally alongside extra_args here would silently shift into
    # mean's slot instead (bit us once already). Keyword-only avoids that
    # footgun regardless of where a given dataset class puts such trailing
    # defaulted params.
    datasets = [
        dataset_cls(path, traj_indices, *extra_args, mean, std, **(extra_kwargs or {}))
        for path, traj_indices in h5_paths_and_indices
    ]
    ds = ConcatDataset(datasets)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=num_workers > 0)
    return ds, loader


def load_euler_data(config, data_root):
    """Emulator loader (windowed k->n_rollout, state prediction). Returns
    (train_loader, val_loader, test_loader, n_train, n_val, n_test,
     n_batch_train, mean, std, stats)."""
    k = config["k"]
    n_rollout = config["n_rollout_train"]
    batch_size = config["batch_size"]
    num_workers = config.get("num_workers", 2)
    stride = config.get("stride", 1)
    frame_skip = config.get("frame_skip", 1)
    prediction_mode = config.get("prediction_mode", "state")

    train_paths = [os.path.join(data_root, f) for f in config["train_files"]]
    val_test_path = os.path.join(data_root, config["val_test_file"])
    val_idx = config["val_traj_idx"]
    test_idx = config["test_traj_idx"]

    train_infos = [_peek_file_info(p) for p in train_paths]
    gammas = [i[0] for i in train_infos]
    assert len(set(gammas)) == 1, f"inconsistent gamma across train files: {list(zip(train_paths, gammas))}"
    ref_shape = train_infos[0][2:]
    for p, info in zip(train_paths, train_infos):
        assert info[2:] == ref_shape, f"{p}: (T,H,W) {info[2:]} != {ref_shape} (incompatible files)"

    n_traj_train = sum(i[1] for i in train_infos)
    print(f"  Train files ({len(train_paths)}) -- gamma={gammas[0]}, {n_traj_train} trajectories total:", flush=True)
    for path, info in zip(train_paths, train_infos):
        print(f"    {path} : {info[1]} trajectories", flush=True)

    gamma_val_test, n_traj_val_test, T_vt, H_vt, W_vt = _peek_file_info(val_test_path)
    print(f"  Val/test file: {val_test_path}", flush=True)
    print(f"  gamma={gamma_val_test}  n_traj={n_traj_val_test}  shape=(T={T_vt}, H={H_vt}, W={W_vt})", flush=True)
    print(f"  Val/test trajectory split -- val:{val_idx}  test:{test_idx}", flush=True)

    assert gammas[0] == gamma_val_test, f"inconsistent gamma between train ({gammas[0]}) and val/test ({gamma_val_test})"
    assert set(val_idx) | set(test_idx) <= set(range(n_traj_val_test)), "trajectory index out of range for the val/test file"
    assert not (set(val_idx) & set(test_idx)), "val/test overlap"

    print("  Computing per-channel normalization stats (streaming, train split only)...", flush=True)
    mean, std = compute_channel_stats_streaming(train_paths)
    print(f"  Per-channel mean ({FIELD_ORDER}): {mean}", flush=True)
    print(f"  Per-channel std  ({FIELD_ORDER}): {std}", flush=True)

    train_sources = [(path, range(info[1])) for path, info in zip(train_paths, train_infos)]
    train_ds, train_loader = _build_lazy_loader(
        LazyEulerDataset, train_sources, (k, n_rollout, stride), mean, std, batch_size, num_workers, shuffle=True,
        extra_kwargs={"frame_skip": frame_skip, "prediction_mode": prediction_mode})
    val_ds, val_loader = _build_lazy_loader(
        LazyEulerDataset, [(val_test_path, val_idx)], (k, n_rollout, stride), mean, std, batch_size, num_workers, shuffle=False,
        extra_kwargs={"frame_skip": frame_skip, "prediction_mode": prediction_mode})
    test_ds, test_loader = _build_lazy_loader(
        LazyEulerDataset, [(val_test_path, test_idx)], (k, n_rollout, stride), mean, std, batch_size, num_workers, shuffle=False,
        extra_kwargs={"frame_skip": frame_skip, "prediction_mode": prediction_mode})

    stats = {
        "gamma": gammas[0],
        "train_files": config["train_files"],
        "val_test_file": config["val_test_file"],
        "field_order": FIELD_ORDER,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_traj_train": n_traj_train,
        "val_traj_idx": val_idx,
        "test_traj_idx": test_idx,
        "frame_skip": frame_skip,
        "prediction_mode": prediction_mode,
    }

    return (train_loader, val_loader, test_loader,
            len(train_ds), len(val_ds), len(test_ds), len(train_loader),
            mean, std, stats)


def load_euler_diffusion_data(config, data_root):
    """Diffusion-corrector loader (single snapshots, no target). Returns
    (train_loader, n_train, n_batch_train, mean, std, stats). Normalization
    stats are computed the same way as the emulator's own (streaming, train
    split only) -- pass the emulator's already-computed mean/std via
    config["field_mean"]/config["field_std"] to avoid a second streaming
    pass over the same files when training the pair back to back."""
    batch_size = config["batch_size"]
    num_workers = config.get("num_workers", 2)
    stride = config.get("stride", 1)

    train_paths = [os.path.join(data_root, f) for f in config["train_files"]]
    train_infos = [_peek_file_info(p) for p in train_paths]
    n_traj_train = sum(i[1] for i in train_infos)

    if "field_mean" in config and "field_std" in config:
        mean = np.array(config["field_mean"], dtype=np.float32)
        std = np.array(config["field_std"], dtype=np.float32)
    else:
        print("  Computing per-channel normalization stats (streaming, train split only)...", flush=True)
        mean, std = compute_channel_stats_streaming(train_paths)

    train_sources = [(path, range(info[1])) for path, info in zip(train_paths, train_infos)]
    train_ds, train_loader = _build_lazy_loader(
        LazyEulerFirstSnapshot, train_sources, (stride,), mean, std, batch_size, num_workers, shuffle=True)

    stats = {"train_files": config["train_files"], "field_order": FIELD_ORDER,
            "mean": mean.tolist(), "std": std.tolist(), "n_traj_train": n_traj_train}

    return train_loader, len(train_ds), len(train_loader), mean, std, stats
