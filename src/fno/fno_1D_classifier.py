"""
1D counterpart of fno_2D_classifier.py::FNO2D_classifier -- the Re-blind
(mono-parameter) diffusion-corrector backbone, for KS instead of Kolmogorov
flow. Same spectral backbone as fno_1D.py::FNO1D (see that file's docstring
for why a degenerate-2D reshape isn't viable), classifier head via
AdaptiveAvgPool1d/AdaptiveMaxPool1d instead of the 2D variants -- adaptive
pooling itself is dimension-agnostic, only the pool class changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fno.fno_2D import FFNN  # dimension-agnostic (plain nn.Linear MLP), reused as-is
from fno.fno_1D import FourierLayer1D, get_meshgrid_1d


class FNO1D_classifier(nn.Module):
    def __init__(self, input_dim, output_dim, modes, width, l, n_layer=4, hidden_proj=None,
                mlp=True, layers_mlp=None, class_mlp_layers=None, n_cat=1000, device="cpu"):
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

        self.hidden_proj = hidden_proj if hidden_proj else width
        self.padding = 0
        self.activation = nn.LeakyReLU()

        self.P = nn.Linear(self.input_dim + 1, self.width)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layer):
            self.layers.append(FourierLayer1D(self.width, self.width, self.modes, l,
                                              mlp=self.mlp, layers_mlp=self.layers_mlp, device=self.device))

        self.n_cat = n_cat
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.class_mlp_layers = class_mlp_layers if class_mlp_layers is not None else [2 * self.width, 4 * self.width, self.n_cat]
        self.classifier = FFNN(self.class_mlp_layers)

        self.Q = nn.Sequential(nn.Linear(self.width, self.hidden_proj), self.activation,
                               nn.Linear(self.hidden_proj, self.output_dim))

    def forward(self, x, predict_class=True):
        # x: (batch_size, Nx, input_dim)
        meshgrid = get_meshgrid_1d(x.shape, self.device)
        x = torch.cat((x, meshgrid), dim=-1)

        x = self.P(x)
        x = x.permute(0, 2, 1)  # (B, Nx, width) -> (B, width, Nx)

        if self.padding != 0:
            x = F.pad(x, [0, self.padding])

        for layer in self.layers:
            x = self.activation(layer(x))

        if self.padding != 0:
            x = x[..., :-self.padding]

        if predict_class:
            # softmax not applied here -- cross-entropy loss already does it.
            # squeeze(-1) explicit (not bare .squeeze()) so a batch size of 1
            # doesn't also collapse the batch dim.
            x_avg_pool = self.avg_pool(x).squeeze(-1)
            x_max_pool = self.max_pool(x).squeeze(-1)
            y = torch.cat([x_avg_pool, x_max_pool], dim=-1)
            y = self.classifier(y)

        x = x.permute(0, 2, 1)  # (B, width, Nx) -> (B, Nx, width)

        # No trailing squeeze on the field output (unlike FNO2D_classifier's
        # .squeeze(-1)): output_dim=1 here is the norm, and the reverse-
        # diffusion update (Diffusion._reverse_diffusion) needs this shaped
        # exactly like x_t, (B, Nx, output_dim), for its elementwise algebra.
        x = self.Q(x)
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

    def class_distribution(self, x):
        meshgrid = get_meshgrid_1d(x.shape, self.device)
        x = torch.cat((x, meshgrid), dim=-1)
        x = self.P(x)
        x = x.permute(0, 2, 1)

        if self.padding != 0:
            x = F.pad(x, [0, self.padding])

        for layer in self.layers:
            x = self.activation(layer(x))

        if self.padding != 0:
            x = x[..., :-self.padding]

        x_avg_pool = self.avg_pool(x).squeeze(-1)
        x_max_pool = self.max_pool(x).squeeze(-1)
        y = torch.cat([x_avg_pool, x_max_pool], dim=-1)
        y = self.classifier(y)
        return nn.Softmax(-1)(y)

    def classifier_cat(self, x):
        return self.class_distribution(x).argmax(-1)
