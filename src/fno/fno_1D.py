"""
1D counterpart of fno_2D.py::FNO2D, for the Kuramoto-Sivashinsky (KS)
equation (a single periodic spatial axis) instead of the 2D Kolmogorov-flow
velocity field. A degenerate 2D call (B, 1, Nx, C) into FNO2D does NOT work:
IntegralKernel2D's spectral weights have a fixed shape (in, out, 2*modes_x,
modes_y), and slicing a size-1 axis by 2*modes_x around its center breaks
(shape mismatch against those fixed weights) -- a real 1D spectral layer is
needed, not a reshape trick (cf. session audit before this file was added).

Mirrors fno_2D.py's structure and naming 1:1 wherever the dimensionality
allows (FFNN unchanged, FFNN2D -> FFNN1D, get_meshgrid -> get_meshgrid_1d,
IntegralKernel2D -> IntegralKernel1D, FourierLayer2D -> FourierLayer1D,
FNO2D -> FNO1D) so the two are easy to compare side by side.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_meshgrid_1d(shape, device="cpu"):
    batch_size, x_size, _ = shape
    x_grid = torch.linspace(0, 1, x_size, device=device)
    x_grid = x_grid.reshape(1, x_size, 1).repeat([batch_size, 1, 1])
    return x_grid.float()


class FFNN1D(nn.Module):
    """Pointwise feed-forward network via Conv1d(kernel_size=1) -- 1D
    counterpart of fno_2D.py::FFNN2D."""

    def __init__(self, layers_width, activation=nn.GELU(), device="cpu"):
        super().__init__()
        self.layers_width = layers_width
        self.depth = len(layers_width)
        self.activation = activation

        layers = []
        for i in range(self.depth - 2):
            layers.append(nn.Conv1d(layers_width[i], layers_width[i + 1], kernel_size=1, device=device))
            layers.append(self.activation)
        layers.append(nn.Conv1d(layers_width[-2], layers_width[-1], kernel_size=1, device=device))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """x: (batch, channels, Nx) -> (batch, channels_out, Nx)"""
        return self.model(x)


class IntegralKernel1D(nn.Module):
    """1D Fourier layer. Unlike IntegralKernel2D (whose 2 spatial axes need
    an rfft + a full-FFT-with-fftshift-and-centering), a single spatial axis
    is exactly analogous to IntegralKernel2D's already-non-negative rfft'd
    axis: just an rfft, keep the first `modes` (low-frequency) coefficients,
    no shift/centering needed."""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        lastdim = x.shape[-1]

        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1], device=x_ft.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        x_out = torch.fft.irfft(out_ft, n=lastdim)
        return x_out


class FourierLayer1D(nn.Module):
    def __init__(self, in_channels, out_channels, modes, l=1, mlp=True, layers_mlp=None, device="cpu"):
        super().__init__()
        self.int_kern = IntegralKernel1D(in_channels, out_channels, modes)
        self.w = nn.Conv1d(in_channels, out_channels, l, padding="same")
        if layers_mlp is not None:
            self.mlp = FFNN1D(layers_mlp, device=device) if mlp else None
        else:
            self.mlp = FFNN1D(3 * [out_channels], device=device)

    def forward(self, x):
        if self.mlp:
            return self.mlp(self.int_kern(x)) + self.w(x)
        return self.int_kern(x) + self.w(x)


class FNO1D(nn.Module):
    def __init__(self, input_dim, output_dim, modes, width, l, n_layer=4, hidden_proj=None,
                mlp=True, layers_mlp=None, device="cpu"):
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

        # Lifting: input channel is input_dim + 1 -- the field plus a single
        # spatial-coordinate channel (vs +2 for 2D's x,y coordinates).
        self.P = nn.Linear(self.input_dim + 1, self.width)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layer):
            self.layers.append(FourierLayer1D(self.width, self.width, self.modes, l,
                                              mlp=self.mlp, layers_mlp=self.layers_mlp, device=self.device))

        self.Q = nn.Sequential(nn.Linear(self.width, self.hidden_proj), self.activation,
                               nn.Linear(self.hidden_proj, self.output_dim))

    def forward(self, x):
        # x: (batch_size, Nx, input_dim)
        meshgrid = get_meshgrid_1d(x.shape, self.device)
        x = torch.cat((x, meshgrid), dim=-1)

        x = self.P(x)
        x = x.permute(0, 2, 1)  # (B, Nx, width) -> (B, width, Nx), for the FFT axis

        if self.padding != 0:
            x = F.pad(x, [0, self.padding])

        for layer in self.layers:
            x = self.activation(layer(x))

        if self.padding != 0:
            x = x[..., :-self.padding]

        x = x.permute(0, 2, 1)  # (B, width, Nx) -> (B, Nx, width)

        # No trailing .squeeze(-1) (unlike FNO2D): output_dim=1 is the norm
        # here (a scalar KS field), and squeezing it away would break the
        # emulator's own x_t + delta shape-matched autoregressive update.
        return self.Q(x)  # (B, Nx, output_dim)

    def count_parameters_per_module(self):
        param_dict = {}
        for name, module in self.named_children():
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            param_dict[name] = trainable
        param_dict["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return param_dict
