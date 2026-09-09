"""
Linear-regression-across-trained-operators reconstruction, cf.
Operator_Identification.ipynb's own markdown cell 0:

    N(u) = N^0(u) + nu * N^1(u)

Given n independently-trained FNO1D operator models (one per nu_i, cf.
training/fno_training_operator_1d.py), each evaluated on the SAME input u,
this is a classic ordinary-least-squares slope/intercept estimate, applied
POINTWISE across every output location (not a single scalar regression):

    N1_hat(u) = sum_i (nu_i - nu_bar)(N_i(u) - N_bar(u)) / sum_i (nu_i - nu_bar)^2
    N0_hat(u) = N_bar(u) - nu_bar * N1_hat(u)

N0_hat/N1_hat can then be combined for ANY nu (including ones none of the
n models were trained on): N_pred(u; nu) = N0_hat(u) + nu * N1_hat(u).

EnsembleFNOOperator is a from-scratch reimplementation of that notebook's
Ensemble_2_fno/Ensemble_n_fno classes (same formula, one class covering both
n=2 and n>2 -- Ensemble_2_fno's own closed form for n=2 is algebraically the
same OLS estimate, cf. Ensemble_n_fno reducing to it at n=2). Fixed one real
bug found while porting: the original Ensemble_n_fno.forward() hardcoded
`.repeat(1, 2001, 50)` -- the exact (T, m) shape of ITS OWN encoded dataset
-- instead of reading N_bar.shape dynamically, so it would silently break
(wrong broadcast, not even a clean error in every case) on any other output
shape. Fixed here to read the shape from the actual tensors at call time.
"""

import torch
import torch.nn as nn


class EnsembleFNOOperator(nn.Module):
    """fno_list[i] is assumed trained at nu_list[i] (cf.
    training/fno_training_operator_1d.py's per-nu models). forward(x)
    returns (N0_hat, N1_hat); combine as N0_hat + nu*N1_hat for any nu."""

    def __init__(self, fno_list, nu_list):
        super().__init__()
        self.fno_list = nn.ModuleList(fno_list)
        self.register_buffer("nu_tensor", torch.tensor([float(n) for n in nu_list]))

    def forward(self, x):
        # x: (B, Mx, 1) -- same input convention as FNO1D.forward
        preds = torch.stack([model(x) for model in self.fno_list], dim=0)  # (n, B, Mx, 1)

        nu = self.nu_tensor.to(x.device)
        nu_bar = nu.mean()
        n_shape = (-1,) + (1,) * (preds.dim() - 1)  # broadcast (n,) against (n, B, Mx, 1)
        nu_dev = (nu - nu_bar).reshape(n_shape)

        n_bar = preds.mean(dim=0)
        n1_hat = (nu_dev * (preds - n_bar)).sum(dim=0) / (nu_dev ** 2).sum(dim=0)
        n0_hat = n_bar - nu_bar * n1_hat
        return n0_hat, n1_hat

    def predict(self, x, nu):
        n0_hat, n1_hat = self(x)
        return n0_hat + nu * n1_hat


@torch.no_grad()
def relative_l2_error(pred, target):
    return (torch.mean((pred - target) ** 2) / torch.mean(target ** 2)).sqrt() * 100
