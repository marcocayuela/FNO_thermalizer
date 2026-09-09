"""
Dataset for KS "operator identification": instead of an autoregressive
next-step emulator, a single FNO1D is trained to map an instantaneous state
u(., t) directly onto the material derivative Du/Dt(., t) -- i.e. to
approximate the KS right-hand side itself,

    Du/Dt = -Delta(u) - nu*Delta^2(u) = N(u)

Reproduces /Users/marco/Documents/PhD/KS_equation/Operator_Identification.ipynb's
own methodology (material-derivative target via finite differences, shuffled
time-snapshots -- no time ordering needed since each sample is an
independent (u, N(u)) pair) on the h5 data this project already generates
(cf. KS_equation/1-D_periodic_Kuramoto_Sivashinsky_equation/
generate_ks_dataset.py), one dataset per nu.

Unlike that notebook, this operates at full spatial resolution (Mx points,
no "grid_encoding" downsampling to m=50 points) -- that encoding existed
there to keep 13 independent trainings cheap on a laptop; not needed given
mesu's GPU budget, and it otherwise trains against a coarsened, structurally
distorted version of the true operator.
"""

import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def material_derivative(u, dt, dx):
    """u: (T, Mx). Returns Du/Dt = du/dt + u*du/dx, (T, Mx) -- same formula
    as the notebook's dudt = gradient(u,dt,time) + 0.5*gradient(u**2,dx,space)
    (0.5*d(u^2)/dx = u*du/dx), just laid out (T, Mx) instead of (Mx, T)."""
    dudt = np.gradient(u, dt, axis=0, edge_order=2)
    dudt = dudt + 0.5 * np.gradient(u ** 2, dx, axis=1, edge_order=2)
    return dudt


class KSOperatorDataset(Dataset):
    """All train_traj snapshots for one nu, each tagged with its own
    (u(.,t), Du/Dt(.,t)) pair -- order-independent (shuffled by the
    DataLoader), matching the notebook's own "shuffle then split" scheme."""

    def __init__(self, exp_root, nu, ds=1, split="train_traj"):
        data_dir = os.path.join(exp_root, f"nu{_format_nu(nu)}", split)
        sim_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
        if not sim_files:
            raise FileNotFoundError(f"No run found for nu={nu} in {data_dir}")

        all_u, all_dudt = [], []
        for path in sim_files:
            with h5py.File(path, "r") as f:
                u = f["state"][:, ::ds, 0].astype(np.float64)  # (T, Mx)
                L, Mx, Tf, nt = f.attrs["L"], f.attrs["Mx"], f.attrs["Tf"], f.attrs["nt"]
            dt = Tf / nt
            dx = L / Mx * ds
            dudt = material_derivative(u, dt, dx)
            all_u.append(u)
            all_dudt.append(dudt)

        self.u = torch.from_numpy(np.concatenate(all_u, axis=0)).float()          # (N, Mx)
        self.dudt = torch.from_numpy(np.concatenate(all_dudt, axis=0)).float()    # (N, Mx)

    def __len__(self):
        return self.u.shape[0]

    def __getitem__(self, idx):
        return self.u[idx], self.dudt[idx]


def _format_nu(nu: float) -> str:
    s = f"{nu:g}"
    return s.replace(".", "p").replace("-", "m")


def build_operator_loaders(data_rep, exp_dir, nu, batch_size, num_workers, ds=1,
                           train_frac=0.7, test_frac=0.3, seed=0):
    """train_frac/test_frac default to 0.7/0.3 (all remaining data used as
    test, matching n_test = N - n_train in the notebook, not its own stale
    "prop_train=0.8" comment which its code never actually uses)."""
    exp_root = os.path.join(data_rep, exp_dir)
    dataset = KSOperatorDataset(exp_root, nu, ds=ds)
    N = len(dataset)
    n_train = int(train_frac * N)
    n_test = N - n_train
    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=max(n_test, 1), shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, n_train, n_test
