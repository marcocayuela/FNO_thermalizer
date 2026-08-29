import glob
import numpy as np
import os
import h5py
import torch

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split
from torch.utils.data import Dataset


def format_re(re: float) -> str:
    s = f"{re:g}"
    return s.replace(".", "p").replace("-", "m")


def compute_field_stats(exp_root, re_values, ds=1, max_frames_per_re=2000):
    """Per-field (u,v) mean/std for the raw state x0, pooled across
    re_values' Re<X>/train_traj/*.h5 files -- used to normalize the
    Re-conditioned diffusion corrector when DatasetManagerMultiRe(...,
    normalize=True). Local counterpart of
    Parameterized_Neural_Operator/training/dataset_kolmogorov.py::
    compute_field_stats (x-only here: the diffusion model has no delta
    target, its target is noise ~N(0,I), independent of x0's own scale).

    Streaming/bounded (not a full torch.cat(all_x) load like
    DatasetManagerMulti's mono-Re x_mean/x_std -- a multi-Re sweep is tens of
    GB, cf. ParametricFirstSnapshot's own lazy-read docstring): each file
    contributes at most max_frames_per_re frames, evenly spaced. Reduction is
    over (frame, H, W), keeping the channel (u,v) dim separate -- same
    convention as DatasetManagerMulti's own all_x.mean(dim=(0,1,2)).
    """
    x_sum = x_sq = None
    n_x = 0

    for re in re_values:
        train_dir = os.path.join(exp_root, f"Re{format_re(re)}", "train_traj")
        for path in sorted(glob.glob(os.path.join(train_dir, "*.h5"))):
            with h5py.File(path, "r") as f:
                n_raw = f["velocity_field"].shape[0]
                n_frames = min(n_raw, max_frames_per_re)
                idx = np.unique(np.linspace(0, n_raw - 1, n_frames).astype(int))
                frames = f["velocity_field"][idx][:, ::ds, ::ds].astype(np.float64)

            flat = frames.reshape(-1, frames.shape[-1])
            x_sum = flat.sum(axis=0) if x_sum is None else x_sum + flat.sum(axis=0)
            x_sq = (flat ** 2).sum(axis=0) if x_sq is None else x_sq + (flat ** 2).sum(axis=0)
            n_x += flat.shape[0]

    x_mean = x_sum / n_x
    x_std = np.sqrt(np.maximum(x_sq / n_x - x_mean ** 2, 1e-12))
    return torch.tensor(x_mean, dtype=torch.float32), torch.tensor(x_std, dtype=torch.float32)



class SequenceDataset(Dataset):
    
    def __init__(self,
                 data,
                 seq_length,
                 stride=1,
                 normalize=False,
                 x_mean=None,
                 x_std=None,
                 y_mean=None,
                 y_std=None,
                 prediction_mode="delta"):
        
        """
        data: array ou tensor de forme (nt, ...)
        seq_length: horizon de prédiction
        stride: pas entre les séquences 
        """
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)

        self.data = data.float()
        self.seq_length = seq_length
        self.stride = stride

        self.normalize = normalize

        self.x_mean = x_mean
        self.x_std = x_std

        self.y_mean = y_mean
        self.y_std = y_std
        self.prediction_mode = prediction_mode

    def __len__(self):
        # on a besoin de t + seq_length
        return (self.data.shape[0] - self.seq_length - 1) // self.stride 

    def __getitem__(self, idx):
        # snapshot initial
        idx = idx * self.stride
        x0 = self.data[idx]

        if self.prediction_mode == "state":
            y = self.data[idx + 1 : idx + 1 + self.seq_length]
        else:
            y = (
                self.data[idx + 1 : idx + 1 + self.seq_length]
                - self.data[idx : idx + self.seq_length]
            )

        if self.normalize:
            x0 = (x0 - self.x_mean) / (self.x_std + 1e-8)
            y = (y - self.y_mean) / (self.y_std + 1e-8)
        
        return x0, y
    
class FirstSnapshot(Dataset):

    def __init__(self, data, seq_length, stride=1):
        """
        data: array ou tensor de forme (nt, ...)
        seq_length: horizon de prédiction
        stride: pas entre les séquences 
        """
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)

        self.data = data.float()
        self.seq_length = seq_length
        self.stride = stride

    def __len__(self):
        # on a besoin de t + seq_length
        return (self.data.shape[0] - self.seq_length - 1) // self.stride 

    def __getitem__(self, idx):
        # snapshot initial
        idx = idx * self.stride
        x0 = self.data[idx]                      # (...)

        # séquence future
        #y = self.data[idx + 1 : idx + 1 + self.seq_length] - self.data[idx : idx + self.seq_length]  # (seq_length, ...)

        return x0#, y


class DatasetManagerMulti():

    def __init__(self, data_rep, exp_dir, seq_length, batch_size, num_workers, ratio=1, train_frac=0.3, test_frac=0.1, stride=1, ds=2, diffusion=False, normalize=True, prediction_mode="delta"):

        self.exp_dir = exp_dir
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_dir = None
        self.training_loader = None
        self.testing_loader = None
        self.stride = stride
        self.ds = ds
        self.prediction_mode = prediction_mode
        if type(ratio)!=int or ratio <1:
            print("Ratio must be an integer greater than 1. It has been automatically set to 1")
            self.ratio = 1
        else:
            self.ratio = ratio  

        if self.exp_dir == "kolmogorov/Re34" or self.exp_dir == "kolmogorov/Re90":

            data_dir = os.path.join(data_rep, self.exp_dir, "train_traj") ### NAME OF THE FOLDER CONTAINING THE TRAINING SIMULATIONS

            simulation_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".h5")]
            
            datasets = []

            if not diffusion:

                all_x = []
                all_y = []
                for sim_file in simulation_files:

                    with h5py.File(sim_file, "r") as f:

                        data = f["velocity_field"][()][::self.ratio, ::self.ds, ::self.ds]
                        tensor_data = torch.from_numpy(data).float()
                        all_x.append(tensor_data[:-1])
                        if self.prediction_mode == "state":
                            all_y.append(tensor_data[1:])
                        else:
                            dx = tensor_data[1:] - tensor_data[:-1]
                            all_y.append(dx)

                all_x = torch.cat(all_x, dim=0)
                all_y = torch.cat(all_y, dim=0)

                x_mean = all_x.mean(dim=(0,1,2))
                x_std = all_x.std(dim=(0,1,2))

                y_mean = all_y.mean(dim=(0,1,2))
                y_std = all_y.std(dim=(0,1,2))
                for sim_file in simulation_files:
                    with h5py.File(sim_file, "r") as f:
                        data = f["velocity_field"][()][::self.ratio,::self.ds, ::self.ds]
                        datasets.append(SequenceDataset(data,
                                                       seq_length=self.seq_length,
                                                       stride=self.stride,
                                                       x_mean=x_mean,
                                                       x_std=x_std,
                                                       y_mean=y_mean,
                                                       y_std=y_std,
                                                       normalize=normalize,
                                                       prediction_mode=self.prediction_mode))
                    
                self.sequence_dataset = ConcatDataset(datasets)

            
                N = len(self.sequence_dataset)
                self.train_frac = train_frac
                self.test_frac = test_frac

                self.n_train = int(self.train_frac*N)
                self.n_test = int(self.test_frac*N)
                self.n_rest = N - self.n_train - self.n_test

                self.training_dataset, self.testing_dataset, _ = random_split(self.sequence_dataset, [self.n_train, self.n_test, self.n_rest])

                self.training_loader = DataLoader(self.training_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=self.num_workers,
                                                pin_memory=False,
                                                persistent_workers=True
                                                )

                self.testing_loader = DataLoader(self.testing_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                num_workers=self.num_workers,
                                                pin_memory=False,
                                                persistent_workers=True
                                                )
                
                self.n_batch_train = len(self.training_loader)
                self.n_batch_test = len(self.testing_loader)


            else:
                for sim_file in simulation_files:
                        with h5py.File(sim_file, "r") as f:
                            data = f["velocity_field"][()][::self.ratio,::self.ds, ::self.ds]
                            datasets.append(FirstSnapshot(data, seq_length=self.seq_length, stride=self.stride))
                        
                self.training_dataset = ConcatDataset(datasets)
                self.n_train = len(self.training_dataset)

                self.training_loader = DataLoader(self.training_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=self.num_workers,
                                                pin_memory=False,
                                                persistent_workers=True
                                                )
                
                self.n_batch_train = len(self.training_loader)


class ParametricFirstSnapshot(Dataset):
    """Like FirstSnapshot, but lazy (h5py opened per-worker, cf.
    Parameterized_Neural_Operator/training/dataset_kolmogorov.py::ParametricSequenceDataset)
    and tags each sample with the Re of the trajectory it comes from -- needed
    to train a single diffusion corrector across several Re instead of one
    model per Re. No target sequence: the diffusion model only needs an
    initial frame to noise/denoise, unlike SequenceDataset's emulator targets.
    """

    def __init__(self, path, re_value, seq_length, stride=1, ds=1):
        self.path = path
        self.re_value = float(re_value)
        self.seq_length = seq_length
        self.stride = stride
        self.ds = ds
        self._file = None

        with h5py.File(path, "r") as f:
            self.n_frames = f["velocity_field"].shape[0]

    def _dataset(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file["velocity_field"]

    def __len__(self):
        return max(0, (self.n_frames - self.seq_length - 1) // self.stride)

    def __getitem__(self, idx):
        idx = idx * self.stride
        x0 = self._dataset()[idx, ::self.ds, ::self.ds]
        x0 = torch.from_numpy(x0).float()
        re = torch.tensor(self.re_value, dtype=torch.float32)
        return x0, re


class DatasetManagerMultiRe():
    """Multi-Re counterpart to DatasetManagerMulti's diffusion branch:
    trains a single diffusion corrector across several Reynolds numbers
    instead of one model per Re. Points at the Re<X>/train_traj/*.h5 layout
    already used by Parameterized_Neural_Operator's kolmogorov_parametric
    dataset (same data, reused as-is -- no separate generation needed).

    Exposes training_loader / n_train / n_batch_train, same attribute names
    as DatasetManagerMulti, so it's a drop-in replacement for the diffusion
    training entrypoint. Also exposes param_mean / param_std (stats of
    log(re_values), same formula as
    Parameterized_Neural_Operator/training/dataset_kolmogorov.py:167-168) for
    the model's Re normalization.
    """

    def __init__(self, data_rep, exp_dir, re_values, seq_length, batch_size, num_workers,
                stride=1, ds=2, normalize=False):
        self.exp_dir = exp_dir
        self.re_values = [float(r) for r in re_values]
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stride = stride
        self.ds = ds

        exp_root = os.path.join(data_rep, exp_dir)

        if normalize:
            self.x_mean, self.x_std = compute_field_stats(exp_root, self.re_values, ds=ds)
        else:
            self.x_mean, self.x_std = torch.tensor(0.0), torch.tensor(1.0)
        datasets = []
        for re in self.re_values:
            train_dir = os.path.join(exp_root, f"Re{format_re(re)}", "train_traj")
            sim_files = sorted(glob.glob(os.path.join(train_dir, "*.h5")))
            if not sim_files:
                raise FileNotFoundError(f"No run found for Re={re} in {train_dir}")
            for path in sim_files:
                datasets.append(ParametricFirstSnapshot(path, re, seq_length=seq_length,
                                                        stride=stride, ds=ds))

        self.training_dataset = ConcatDataset(datasets)
        self.n_train = len(self.training_dataset)

        self.training_loader = DataLoader(self.training_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=self.num_workers,
                                          pin_memory=False,
                                          persistent_workers=self.num_workers > 0)

        self.n_batch_train = len(self.training_loader)

        log_re = np.log(np.array(self.re_values))
        self.param_mean = float(log_re.mean())
        self.param_std = float(log_re.std() + 1e-8)


class DatasetManagerKS1D():
    """Mono-parameter counterpart of DatasetManagerMulti, for a 1D
    Kuramoto-Sivashinsky field instead of the 2D Kolmogorov velocity field --
    single spatial axis throughout (data["state"][::ratio, ::ds], reduction
    dims (0,1) instead of (0,1,2)), h5 key "state" instead of
    "velocity_field" (cf. generate_ks_dataset.py). No exp_dir gate (unlike
    DatasetManagerMulti's kolmogorov/Re34-or-Re90 check): this class is only
    ever used for one KS_equation/nu<X> directory at a time, so the branch
    always applies. Reuses SequenceDataset/FirstSnapshot as-is (already
    shape-agnostic, only index along axis 0).
    """

    def __init__(self, data_rep, exp_dir, seq_length, batch_size, num_workers, ratio=1,
                train_frac=0.7, test_frac=0.1, stride=1, ds=1, diffusion=False,
                normalize=True, prediction_mode="delta"):

        self.exp_dir = exp_dir
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stride = stride
        self.ds = ds
        self.prediction_mode = prediction_mode
        if type(ratio) != int or ratio < 1:
            print("Ratio must be an integer greater than 1. It has been automatically set to 1")
            self.ratio = 1
        else:
            self.ratio = ratio

        data_dir = os.path.join(data_rep, self.exp_dir, "train_traj")
        simulation_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".h5")]
        datasets = []

        if not diffusion:
            all_x, all_y = [], []
            for sim_file in simulation_files:
                with h5py.File(sim_file, "r") as f:
                    data = f["state"][()][::self.ratio, ::self.ds]
                    tensor_data = torch.from_numpy(data).float()
                    all_x.append(tensor_data[:-1])
                    if self.prediction_mode == "state":
                        all_y.append(tensor_data[1:])
                    else:
                        all_y.append(tensor_data[1:] - tensor_data[:-1])

            all_x = torch.cat(all_x, dim=0)
            all_y = torch.cat(all_y, dim=0)

            x_mean = all_x.mean(dim=(0, 1))
            x_std = all_x.std(dim=(0, 1))
            y_mean = all_y.mean(dim=(0, 1))
            y_std = all_y.std(dim=(0, 1))

            for sim_file in simulation_files:
                with h5py.File(sim_file, "r") as f:
                    data = f["state"][()][::self.ratio, ::self.ds]
                    datasets.append(SequenceDataset(data, seq_length=self.seq_length, stride=self.stride,
                                                   x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                                                   normalize=normalize, prediction_mode=self.prediction_mode))

            self.sequence_dataset = ConcatDataset(datasets)
            N = len(self.sequence_dataset)
            self.train_frac = train_frac
            self.test_frac = test_frac
            self.n_train = int(self.train_frac * N)
            self.n_test = int(self.test_frac * N)
            self.n_rest = N - self.n_train - self.n_test

            self.training_dataset, self.testing_dataset, _ = random_split(
                self.sequence_dataset, [self.n_train, self.n_test, self.n_rest])

            self.training_loader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True,
                                              num_workers=self.num_workers, pin_memory=False,
                                              persistent_workers=self.num_workers > 0)
            self.testing_loader = DataLoader(self.testing_dataset, batch_size=self.batch_size, shuffle=False,
                                             num_workers=self.num_workers, pin_memory=False,
                                             persistent_workers=self.num_workers > 0)
            self.n_batch_train = len(self.training_loader)
            self.n_batch_test = len(self.testing_loader)

        else:
            for sim_file in simulation_files:
                with h5py.File(sim_file, "r") as f:
                    data = f["state"][()][::self.ratio, ::self.ds]
                    datasets.append(FirstSnapshot(data, seq_length=self.seq_length, stride=self.stride))

            self.training_dataset = ConcatDataset(datasets)
            self.n_train = len(self.training_dataset)
            self.training_loader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True,
                                              num_workers=self.num_workers, pin_memory=False,
                                              persistent_workers=self.num_workers > 0)
            self.n_batch_train = len(self.training_loader)



