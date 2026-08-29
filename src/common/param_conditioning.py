"""
FiLM conditioning on a scalar physical parameter (e.g. Reynolds number).

Local copy of Parameterized_Neural_Operator/models/film.py's ParamEncoder/FiLM
-- not imported cross-repo on purpose: this project's own training/ package
(a regular package, has __init__.py) silently shadows PNO's training/ the
moment both are on sys.path together, so PNO's training.DiffusionModel
becomes unreachable regardless of import order (see
hyper_r_correction_multi_re.py's docstring in the PNO repo for the full
explanation). Rather than have thermalizer's *production* models depend on
PNO's codebase (fragile for the same reason), only cross-repo analysis
scripts import across the boundary -- the models themselves stay
self-contained.

ParamEncoder turns the parameter into an embedding vector; FiLM applies a
per-channel affine modulation from that embedding.
"""

import torch
import torch.nn as nn


class ParamEncoder(nn.Module):

    def __init__(self, embed_dim, hidden_dim=64, n_layers=2, log_transform=True,
                 param_mean=0.0, param_std=1.0):
        super().__init__()
        self.log_transform = log_transform
        self.register_buffer("param_mean", torch.tensor(float(param_mean)))
        self.register_buffer("param_std", torch.tensor(float(param_std)))

        layers = [nn.Linear(1, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, embed_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, param):
        # param: (B,) raw values (e.g. Re) -> (B, embed_dim)
        param = param.reshape(-1, 1).float()
        if self.log_transform:
            param = torch.log(param)
        param = (param - self.param_mean) / self.param_std
        return self.mlp(param)


class FiLM(nn.Module):

    """Per-channel affine modulation: x -> (1 + gamma(cond)) * x + beta(cond).

    The projection to (gamma, beta) is zero-initialized so the layer starts
    out as the identity (the network learns the dependence on the parameter
    progressively rather than starting from a random modulation).
    """

    def __init__(self, embed_dim, n_channels):
        super().__init__()
        self.to_gamma_beta = nn.Linear(embed_dim, 2 * n_channels)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x, cond_embed):
        # x: (B, C, H, W); cond_embed: (B, embed_dim)
        gamma, beta = self.to_gamma_beta(cond_embed).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * x + beta
