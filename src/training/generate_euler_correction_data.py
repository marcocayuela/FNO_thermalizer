"""
Generates training data for the residual corrector (cf.
training/ResidualCorrectorModel.py, training/trainer_residual_corrector.py):
runs the FROZEN, already-trained delta-mode emulator (e.g.
fno_euler_gamma1.4_delta) autoregressively on selected GT trajectories, and
saves ONLY the predicted rollout states to a new, lightweight HDF5 file.

Why only the predicted states: the corresponding TRUE state at step s is
already sitting in the original raw Euler HDF5 file at time index s (frame
0 of the rollout IS the true GT frame 0) -- duplicating it would roughly
double storage for no reason. training/euler_dataset.py::LazyEulerCorrectionDataset
reads the true frame from the original file and the predicted frame from
this generated file, paired by (source_file, source_traj_idx, step).

Why this data, not synthetic noise: the previous corrector
(diffusion_euler_gamma1.4) was trained to remove synthetic Gaussian noise
from real snapshots, and empirically could not stabilize the delta
emulator's rollout (it even slightly worsened it) -- its own error mode
(a real, deterministic, exponentially-growing numerical instability) looks
nothing like additive noise. This script captures that ACTUAL corruption
directly by recording (emulator's own prediction at step s, s) for real
rollouts, from step 0 (clean) through well past where the emulator is known
to diverge (~150-300 steps, cf. this session's own correction_eval_euler.py
runs on fno_euler_gamma1.4_delta).

Trajectory selection: pass --emulator_run so this script can read that
run's own config.yaml (train_files, val_test_file, val_traj_idx, mean, std)
directly -- no need to respecify paths. --role selects which pool to draw
from:
    train  -> up to --n_traj_per_file trajectories from EACH of the
              emulator's own train_files (already seen by the emulator --
              this is fine, the corrector is learning that emulator's own
              error behavior, not being tested for generalization here)
    val    -> the emulator's own val_traj_idx trajectories, from its
              val_test_file
--role val is meant to become the corrector's OWN held-out test split
(Tr/Te tracking during its training, cf. trainer_residual_corrector.py) --
test_traj_idx is deliberately never touched by this script at all, reserved
for the final, fully-held-out rollout-correction check
(evaluation/correction_eval_euler.py or its residual-corrector counterpart).

Usage (from thermalizer/src, needs a GPU for a reasonable runtime -- this
is pure inference, no backprop, but still ~n_traj * n_steps forward passes):
    python training/generate_euler_correction_data.py \\
        --emulator_run $LOG_DIR/euler_multi_quadrants_openBC/fno_euler_gamma1.4_delta \\
        --role train --n_traj_per_file 10 --n_steps 300 \\
        --out_file $DATA_DIR/euler_multi_quadrants_openBC/correction_data/train_rollouts.hdf5

    python training/generate_euler_correction_data.py \\
        --emulator_run $LOG_DIR/euler_multi_quadrants_openBC/fno_euler_gamma1.4_delta \\
        --role val --n_steps 300 \\
        --out_file $DATA_DIR/euler_multi_quadrants_openBC/correction_data/val_rollouts.hdf5
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.correction_eval import load_config, load_checkpoint_dict
from training.euler_dataset import N_CHANNELS, _peek_file_info
from training.fno_training_euler import EmulatorFNO


def load_emulator_euler(run_dir, device):
    """Mirrors evaluation/correction_eval_euler.py::load_emulator_fno_euler
    (not imported directly from there to avoid a circular import --
    correction_eval_euler.py is an evaluation/ script, this is a
    training/ one)."""
    cfg = load_config(run_dir)
    model = EmulatorFNO(
        input_dim=cfg["n_channels"] * cfg["k"], output_dim=cfg["n_channels"],
        modes_x=cfg["modes_x"], modes_y=cfg["modes_y"], width=cfg["width"], l=cfg["kernel_size"],
        n_layer=cfg["n_layers"], hidden_proj=cfg.get("hidden_proj"), mlp=cfg.get("mlp", True),
        layers_mlp=cfg.get("layers_mlp"), tau=cfg.get("tau", 1e-5), padding=cfg.get("padding", 0),
        device=device,
    )
    ckpt = load_checkpoint_dict(run_dir, ["min_test_loss.pth", "final_model.pth", "min_train_loss.pth"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).float().eval(), cfg


@torch.no_grad()
def rollout_predicted_only(model, prediction_mode, tau, x0, n_steps, device):
    """x0: (H, W, C) raw (un-normalized) GT frame 0. Returns (n_steps+1, H, W, C)
    float32 numpy array, frame 0 == x0 itself, frames 1..n_steps the
    emulator's own free-running prediction -- mirrors
    evaluation/correction_eval.py::rollout()'s core loop exactly (same
    delta/state branching, same tau*randn_like perturbation) but without
    any diffusion correction or early-stopping, since here we deliberately
    WANT the full trajectory including well past divergence."""
    H, W, C = x0.shape
    traj = torch.empty((n_steps + 1, H, W, C), dtype=torch.float32)
    x_t = x0.unsqueeze(0).to(device)
    traj[0] = x0

    for i in range(n_steps):
        if prediction_mode == "state":
            x_t = model(x_t)
        else:
            x_dt = model(x_t)
            x_t = x_t + x_dt + tau * torch.randn_like(x_dt)
        traj[i + 1] = x_t[0].cpu()

    return traj.numpy()


def load_raw_frame0(h5_path, traj_idx, mean, std):
    """Loads and normalizes just frame 0 of one trajectory -- the emulator's
    own rollout starting point, in the SAME normalized space it was trained
    in (cf. training/euler_dataset.py::LazyEulerDataset)."""
    with h5py.File(h5_path, "r") as f:
        energy = f["t0_fields"]["energy"][traj_idx, 0]
        density = f["t0_fields"]["density"][traj_idx, 0]
        momentum = f["t1_fields"]["momentum"][traj_idx, 0]
    frame = np.empty((*energy.shape, N_CHANNELS), dtype=np.float32)
    frame[..., 0] = energy
    frame[..., 1] = density
    frame[..., 2:4] = momentum
    return torch.from_numpy((frame - mean) / std).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=os.environ.get("DATA_DIR", "../data"))
    parser.add_argument("--emulator_run", required=True)
    parser.add_argument("--role", choices=["train", "val"], required=True)
    parser.add_argument("--n_traj_per_file", type=int, default=10,
                        help="only used for --role train: cap on how many trajectories to draw "
                        "from EACH of the emulator's own train_files")
    parser.add_argument("--n_steps", type=int, default=300)
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                              else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    print(f"Loading frozen emulator from {args.emulator_run}...", flush=True)
    model, cfg = load_emulator_euler(args.emulator_run, device)
    prediction_mode = cfg.get("prediction_mode", "state")
    mean = np.array(cfg["field_mean"], dtype=np.float32)
    std = np.array(cfg["field_std"], dtype=np.float32)
    print(f"  prediction_mode={prediction_mode}  frame_skip={cfg.get('frame_skip', 1)}", flush=True)

    # (source_file relative path, traj_idx) pairs to roll out.
    if args.role == "train":
        sources = []
        for rel_path in cfg["train_files"]:
            full_path = os.path.join(args.data_dir, rel_path)
            _, n_traj, _, _, _ = _peek_file_info(full_path)
            chosen = list(range(min(args.n_traj_per_file, n_traj)))
            sources += [(rel_path, i) for i in chosen]
    else:
        sources = [(cfg["val_test_file"], i) for i in cfg["val_traj_idx"]]

    print(f"  {len(sources)} trajectories to roll out ({args.n_steps} steps each):", flush=True)
    for rel_path, traj_idx in sources:
        print(f"    {rel_path} [traj {traj_idx}]", flush=True)

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with h5py.File(args.out_file, "w") as out:
        out.attrs["emulator_run"] = args.emulator_run
        out.attrs["prediction_mode"] = prediction_mode
        out.attrs["n_steps"] = args.n_steps
        out.attrs["field_mean"] = mean
        out.attrs["field_std"] = std
        # frame_skip of the SOURCE data this rollout advances through one
        # native step at a time (this emulator's own training frame_skip,
        # cf. training/euler_dataset.py::LazyEulerDataset) -- so
        # predicted[traj, s] lines up with the true file's frame index
        # s * frame_skip, not bare s, if frame_skip != 1.
        out.attrs["frame_skip"] = cfg.get("frame_skip", 1)
        # Per-row provenance -- which original file/trajectory each row of
        # "predicted" corresponds to, so LazyEulerCorrectionDataset can
        # fetch the matching TRUE frame straight from that original file
        # without any separate config bookkeeping at dataset-construction time.
        dt = h5py.special_dtype(vlen=str)
        source_files_ds = out.create_dataset("source_files", (len(sources),), dtype=dt)
        source_traj_idx_ds = out.create_dataset("source_traj_idx", (len(sources),), dtype=np.int64)

        predicted_ds = None
        for row, (rel_path, traj_idx) in enumerate(sources):
            full_path = os.path.join(args.data_dir, rel_path)
            print(f"  [{row + 1}/{len(sources)}] rolling out {rel_path} [traj {traj_idx}]...", flush=True)
            x0 = load_raw_frame0(full_path, traj_idx, mean, std)
            traj_pred = rollout_predicted_only(model, prediction_mode, model.tau, x0, args.n_steps, device)

            if predicted_ds is None:
                predicted_ds = out.create_dataset(
                    "predicted", shape=(len(sources), *traj_pred.shape), dtype=np.float32,
                    chunks=(1, 1, *traj_pred.shape[1:]),
                )
            predicted_ds[row] = traj_pred
            source_files_ds[row] = rel_path
            source_traj_idx_ds[row] = traj_idx

    size_gb = os.path.getsize(args.out_file) / 1e9
    print(f"\nDone. Wrote {args.out_file} ({size_gb:.2f} GB).", flush=True)


if __name__ == "__main__":
    main()
