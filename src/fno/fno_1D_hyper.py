"""
1D counterpart of fno_2D_classifier_hyper.py's hyper-R mechanism, applied to
a plain EMULATOR (no classifier head) instead of the diffusion corrector's
backbone -- direct port of Parameterized_Neural_Operator/models/pfno_hyper_2D.py
::PFNO2DHyper to a single periodic spatial axis, for the Kuramoto-Sivashinsky
(KS) equation conditioned on its viscosity-like parameter nu instead of
Kolmogorov flow's Reynolds number.

Mirrors fno_1D.py's own naming 1:1 wherever the mechanism carries over
(IntegralKernel1D -> HyperIntegralKernel1D, FourierLayer1D ->
HyperFourierLayer1D, FNO1D -> FNO1D_hyper), the same way fno_1D.py itself
mirrors fno_2D.py -- so all three (fno_1D, fno_1D_hyper, fno_2D_classifier_hyper)
stay easy to compare side by side.

Unlike EmulatorFNO1D/FNO1D (cf. training/fno_training_1d.py's own docstring:
"no denormalize buffer... catastrophic for KS, whose delta std over one dt
is much smaller than the field's own std"), this model carries its own
x_mean/x_std/y_mean/y_std buffers (same mechanism as PFNO2DHyper) -- input
normalized going in, output denormalized coming back out, always paired and
persisted in the checkpoint, so the exact footgun that docstring warns about
cannot happen here regardless of how the caller's dataset does or doesn't
normalize.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.param_conditioning import ParamEncoder
from fno.fno_1D import FFNN1D, get_meshgrid_1d


class HyperIntegralKernel1D(nn.Module):
    """Spectral weight R parameterized by nu via a learned basis mixture:
    R(nu) = sum_k a_k(nu) * R_k. Direct 1D port of fno_2D_classifier_hyper.py
    ::HyperIntegralKernel2D -- see that file's docstring for the full
    rationale. a_k(nu) starts at a constant 1/n_basis (zero-init weight,
    bias=1/n_basis), so R(nu) starts as a plain nu-independent average of
    the basis matrices."""

    def __init__(self, in_channels, out_channels, modes, n_basis, embed_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.n_basis = n_basis

        scale = 1 / (in_channels * out_channels)
        self.basis_weights = nn.Parameter(
            scale * torch.rand(n_basis, in_channels, out_channels, modes, dtype=torch.cfloat)
        )
        self.to_coeffs = nn.Linear(embed_dim, n_basis)
        nn.init.zeros_(self.to_coeffs.weight)
        nn.init.constant_(self.to_coeffs.bias, 1.0 / n_basis)

    def forward(self, x, cond):
        batchsize = x.shape[0]
        lastdim = x.shape[-1]

        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1], device=x_ft.device, dtype=torch.cfloat)
        modes_part = x_ft[:, :, :self.modes]

        coeffs = self.to_coeffs(cond).to(torch.cfloat)          # (B, n_basis)
        R = torch.einsum("bk,kiox->biox", coeffs, self.basis_weights)  # (B, in, out, modes)
        mixed = torch.einsum("bix,biox->box", modes_part, R)

        out_ft[:, :, :self.modes] = mixed
        return torch.fft.irfft(out_ft, n=lastdim)


class HyperFourierLayer1D(nn.Module):
    def __init__(self, in_channels, out_channels, modes, n_basis, embed_dim,
                 l=1, mlp=True, layers_mlp=None, device="cpu"):
        super().__init__()
        self.int_kern = HyperIntegralKernel1D(in_channels, out_channels, modes, n_basis, embed_dim)
        # W (the local/skip path) stays FIXED, not nu-dependent -- same
        # rationale as pfno_hyper_2D.py/fno_2D_classifier_hyper.py: isolates
        # whether conditioning the spectral operator directly is enough.
        self.w = nn.Conv1d(in_channels, out_channels, l, padding="same")
        if mlp:
            if layers_mlp is not None:
                widths = [out_channels] + list(layers_mlp)[1:-1] + [out_channels]
            else:
                widths = [out_channels, 2 * out_channels, out_channels]
            self.mlp = FFNN1D(widths, device=device)
        else:
            self.mlp = None

    def forward(self, x, cond):
        if self.mlp is not None:
            return self.mlp(self.int_kern(x, cond)) + self.w(x)
        return self.int_kern(x, cond) + self.w(x)


class FNO1D_hyper(nn.Module):
    """FNO1D (cf. fno_1D.py) with the spectral weight R parameterized by nu
    via a learned basis mixture, for a single hyper-nu-conditioned emulator
    trained across several KS viscosities at once instead of one model per
    nu (cf. training/fno_training_1d_hyper.py).

    forward(x, nu) -> (B, Nx, output_dim), already denormalized -- x itself
    is expected in RAW (unnormalized) units; normalize/denormalize both
    happen inside forward() via the x_mean/x_std/y_mean/y_std buffers.
    """

    def __init__(self, input_dim, output_dim, modes, width, l, n_layer=4, hidden_proj=None,
                mlp=True, layers_mlp=None,
                param_embed_dim=32, param_hidden_dim=64, param_encoder_layers=2,
                param_log_transform=True, param_mean=0.0, param_std=1.0,
                n_basis=4, device="cpu",
                x_mean=0.0, x_std=1.0, y_mean=0.0, y_std=1.0):
        super().__init__()
        self.device = device
        self.modes = modes
        self.width = width
        self.l = l
        self.n_layer = n_layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = mlp
        self.layers_mlp = layers_mlp
        self.n_basis = n_basis
        self.hidden_proj = hidden_proj if hidden_proj else width
        self.padding = 0
        self.activation = nn.LeakyReLU()

        # Field normalization -- identity by default (0/1), same mechanism
        # and rationale as DiffusionModel.py::Diffusion's x_mean/x_std (cf.
        # that file's own comment): registered as buffers so they persist in
        # the checkpoint, and a bare (dim,) shape broadcasts against the
        # trailing channel axis regardless of the input's rank.
        def _as_tensor(v, n):
            t = torch.as_tensor(v, dtype=torch.float32)
            return t.expand(n).clone() if t.ndim == 0 else t
        self.register_buffer("x_mean", _as_tensor(x_mean, input_dim))
        self.register_buffer("x_std", _as_tensor(x_std, input_dim))
        self.register_buffer("y_mean", _as_tensor(y_mean, output_dim))
        self.register_buffer("y_std", _as_tensor(y_std, output_dim))

        self.param_encoder = ParamEncoder(
            embed_dim=param_embed_dim, hidden_dim=param_hidden_dim,
            n_layers=param_encoder_layers, log_transform=param_log_transform,
            param_mean=param_mean, param_std=param_std,
        )

        self.P = nn.Linear(input_dim + 1, width)
        self.layers = nn.ModuleList([
            HyperFourierLayer1D(width, width, modes, n_basis, param_embed_dim,
                                l=l, mlp=mlp, layers_mlp=layers_mlp, device=device)
            for _ in range(n_layer)
        ])
        self.Q = nn.Sequential(nn.Linear(width, self.hidden_proj), self.activation,
                               nn.Linear(self.hidden_proj, output_dim))

    def forward(self, x, nu):
        # x: (B, Nx, input_dim) RAW units; nu: (B,) raw nu values
        cond = self.param_encoder(nu.to(x.device))

        x = (x - self.x_mean) / self.x_std

        meshgrid = get_meshgrid_1d(x.shape, self.device)
        x = torch.cat((x, meshgrid), dim=-1)
        x = self.P(x)
        x = x.permute(0, 2, 1)  # (B, width, Nx)

        if self.padding != 0:
            x = F.pad(x, [0, self.padding])
        for layer in self.layers:
            x = self.activation(layer(x, cond))
        if self.padding != 0:
            x = x[..., :-self.padding]

        x = x.permute(0, 2, 1)  # (B, Nx, width)
        out = self.Q(x)  # (B, Nx, output_dim), normalized-target space
        return out * self.y_std + self.y_mean

    def count_parameters_per_module(self):
        d = {name: sum(p.numel() for p in m.parameters() if p.requires_grad)
             for name, m in self.named_children()}
        d["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return d
