import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """2x (Conv2d circular-padded + GroupNorm + GELU). Padding circulaire pour respecter
    la periodicite du domaine de Kolmogorov flow (equivalent convolutif du mode
    'periodization' utilise par WNO)."""

    def __init__(self, in_ch, out_ch, device="cpu"):
        super().__init__()
        n_groups = min(8, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode="circular", device=device),
            nn.GroupNorm(n_groups, out_ch, device=device),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode="circular", device=device),
            nn.GroupNorm(n_groups, out_ch, device=device),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downsampling appris (conv stride 2) + DoubleConv."""

    def __init__(self, in_ch, out_ch, device="cpu"):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, padding_mode="circular", device=device),
            nn.GELU(),
            DoubleConv(in_ch, out_ch, device=device),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """Upsampling (ConvTranspose2d) + concat skip connection + DoubleConv."""

    def __init__(self, in_ch, out_ch, device="cpu"):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2, device=device)
        self.conv = DoubleConv(in_ch, out_ch, device=device)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """U-Net classique pour l'emulation de champs 2D periodiques (Kolmogorov flow).

    Entree/sortie au format channels-last (batch, H, W, C), comme FNO2D/WNO2d, pour
    rester compatible avec DatasetManagerMulti/Trainer sans les modifier.
    """

    def __init__(self, input_dim, output_dim, depth=3, base_width=32, device="cpu"):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.base_width = base_width

        channels = [base_width * (2 ** i) for i in range(depth + 1)]

        self.inc = DoubleConv(input_dim, channels[0], device=device)
        self.downs = nn.ModuleList([
            Down(channels[i], channels[i + 1], device=device) for i in range(depth)
        ])
        self.ups = nn.ModuleList([
            Up(channels[depth - i], channels[depth - i - 1], device=device) for i in range(depth)
        ])
        self.outc = nn.Conv2d(channels[0], output_dim, kernel_size=1, device=device)

    def forward(self, x):
        # (batch, H, W, C) -> (batch, C, H, W)
        x = x.permute(0, 3, 1, 2)

        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs[:-1]:
            x = down(x)
            skips.append(x)
        x = self.downs[-1](x)  # bottleneck

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        x = self.outc(x)
        # (batch, C, H, W) -> (batch, H, W, C)
        return x.permute(0, 2, 3, 1)

    def count_parameters_per_module(self):
        param_dict = {}
        for name, module in self.named_children():
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            param_dict[name] = trainable
        param_dict["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return param_dict
