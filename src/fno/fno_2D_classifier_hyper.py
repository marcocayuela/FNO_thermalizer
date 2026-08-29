import torch
import torch.nn as nn
import torch.nn.functional as F

from common.param_conditioning import ParamEncoder
from fno.fno_2D import FFNN, FFNN2D, get_meshgrid


class HyperIntegralKernel2D(nn.Module):
    """Spectral weight R parameterized by Re via a learned basis mixture:
    R(Re) = sum_k a_k(Re) * R_k. Direct port of
    Parameterized_Neural_Operator/models/pfno_hyper_2D.py::HyperIntegralKernel2D
    (see that file's docstring for the full rationale). a_k(Re) starts at a
    constant 1/n_basis (zero-init weight, bias=1/n_basis), so R(Re) starts as
    a plain Re-independent average of the basis matrices."""

    def __init__(self, in_channels, out_channels, modes_x, modes_y, n_basis, embed_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.n_basis = n_basis

        scale = 1 / (in_channels * out_channels)
        self.basis_weights = nn.Parameter(
            scale * torch.rand(n_basis, in_channels, out_channels, 2 * modes_x, modes_y, dtype=torch.cfloat)
        )
        self.to_coeffs = nn.Linear(embed_dim, n_basis)
        nn.init.zeros_(self.to_coeffs.weight)
        nn.init.constant_(self.to_coeffs.bias, 1.0 / n_basis)

    def forward(self, x, cond):
        lastdim = x.shape[-1]
        x_ft = torch.fft.rfft2(x)
        x_ft = torch.fft.fftshift(x_ft, dim=-2)
        out_ft = torch.zeros(x.shape[0], self.out_channels, x_ft.shape[-2], x_ft.shape[-1],
                             device=x_ft.device, dtype=torch.cfloat)
        midX = x_ft.shape[-2] // 2
        modes = x_ft[:, :, midX - self.modes_x:midX + self.modes_x, :self.modes_y]

        coeffs = self.to_coeffs(cond).to(torch.cfloat)
        R = torch.einsum("bk,kioxy->bioxy", coeffs, self.basis_weights)
        mixed = torch.einsum("bixy,bioxy->boxy", modes, R)

        out_ft[:, :, midX - self.modes_x:midX + self.modes_x, :self.modes_y] = mixed
        out_ft = torch.fft.fftshift(out_ft, dim=-2)
        return torch.fft.irfft2(out_ft, s=(out_ft.shape[-2], lastdim))


class HyperFourierLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y, n_basis, embed_dim,
                 l=1, mlp=True, layers_mlp=None, device="cpu"):
        super().__init__()
        self.int_kern = HyperIntegralKernel2D(in_channels, out_channels, modes_x, modes_y, n_basis, embed_dim)
        # W (the local/skip path) stays FIXED, not Re-dependent -- same
        # rationale as pfno_hyper_2D.py: isolates whether conditioning the
        # spectral operator directly is enough on its own.
        self.w = nn.Conv2d(in_channels, out_channels, l, padding="same")
        # Widths construction matches pfno_hyper_2D.py::HyperFourierLayer2D
        # exactly (not fno_2D.py's own FourierLayer2D, which has a subtly
        # different -- and layers_mlp-ignoring-mlp-flag -- convention).
        if mlp:
            if layers_mlp is not None:
                widths = [out_channels] + list(layers_mlp)[1:-1] + [out_channels]
            else:
                widths = [out_channels, 2 * out_channels, out_channels]
            self.mlp = FFNN2D(widths, device=device)
        else:
            self.mlp = None

    def forward(self, x, cond):
        if self.mlp is not None:
            return self.mlp(self.int_kern(x, cond)) + self.w(x)
        return self.int_kern(x, cond) + self.w(x)


class FNO2D_classifier_hyper(nn.Module):
    """FNO2D_classifier (cf. fno_2D_classifier.py) with the spectral weight R
    parameterized by Re via a learned basis mixture (hyper-R technique, cf.
    HyperIntegralKernel2D above) instead of FiLM or input concatenation --
    direct port of Parameterized_Neural_Operator's pfno_hyper_2D.py, the
    conditioning mechanism found most robust to long-rollout instability in
    that project's own ablation study this session.

    forward(x, re, predict_class=True) -> (pred, class_logits) if
    predict_class, else pred alone (same contract as the other diffusion
    backbones, required by training/DiffusionModel.py:Diffusion).
    """

    def __init__(self, input_dim, output_dim, modes_x, modes_y, width, l, n_layer=4,
                 hidden_proj=None, mlp=True, layers_mlp=None,
                 param_embed_dim=32, param_hidden_dim=64, param_encoder_layers=2,
                 param_log_transform=True, param_mean=0.0, param_std=1.0,
                 n_basis=4, class_mlp_layers=None, n_cat=1000, device="cpu"):
        super().__init__()
        self.device = device
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.width = width
        self.l = l
        self.n_layer = n_layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = mlp
        self.layers_mlp = layers_mlp
        self.n_basis = n_basis
        self.n_cat = n_cat

        if not hidden_proj:
            self.hidden_proj = self.width
        else:
            self.hidden_proj = hidden_proj

        self.padding = 0
        self.activation = nn.LeakyReLU()

        self.param_encoder = ParamEncoder(
            embed_dim=param_embed_dim, hidden_dim=param_hidden_dim,
            n_layers=param_encoder_layers, log_transform=param_log_transform,
            param_mean=param_mean, param_std=param_std,
        )

        self.P = nn.Linear(self.input_dim + 2, self.width)
        self.layers = nn.ModuleList([
            HyperFourierLayer2D(width, width, modes_x, modes_y, n_basis, param_embed_dim,
                                l=l, mlp=mlp, layers_mlp=layers_mlp, device=device)
            for _ in range(n_layer)
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        if class_mlp_layers is None:
            class_mlp_layers = [2 * self.width + 1, 4 * self.width, self.n_cat]
        self.class_mlp_layers = class_mlp_layers
        self.classifier = FFNN(class_mlp_layers)

        self.Q = nn.Sequential(nn.Linear(self.width, self.hidden_proj), self.activation,
                               nn.Linear(self.hidden_proj, self.output_dim))

    def _normalize_param(self, re):
        re = re.reshape(-1, 1).float()
        if self.param_encoder.log_transform:
            re = torch.log(re)
        return (re - self.param_encoder.param_mean) / self.param_encoder.param_std

    def forward(self, x, re, predict_class=True):
        re = re.to(x.device)
        cond = self.param_encoder(re)  # (B, embed_dim)

        meshgrid = get_meshgrid(x.shape, self.device)
        x = torch.cat((x, meshgrid), dim=-1)
        x = self.P(x)
        x = x.permute(0, 3, 1, 2)

        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for layer in self.layers:
            x = self.activation(layer(x, cond))

        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]

        if predict_class:
            avg = self.avg_pool(x).flatten(1)
            mx = self.max_pool(x).flatten(1)
            re_scalar = self._normalize_param(re)
            y = torch.cat([avg, mx, re_scalar], dim=-1)
            y = self.classifier(y)

        x = x.permute(0, 2, 3, 1)
        x = self.Q(x).squeeze(-1)

        if predict_class:
            return x, y
        return x

    def count_parameters_per_module(self):
        param_dict = {}
        for name, module in self.named_children():
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            param_dict[name] = trainable
        param_dict["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return param_dict
