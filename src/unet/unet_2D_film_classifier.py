import torch
import torch.nn as nn

from common.param_conditioning import ParamEncoder, FiLM
from fno.fno_2D import FFNN


class DoubleConvFiLM(nn.Module):
    """DoubleConv (cf. unet_2D.py) + a FiLM modulation on the block's output,
    conditioned on Re. One FiLM instance per resolution level (channel counts
    differ per level), applied AFTER the block's own GroupNorm+GELU stack --
    zero-initialized, so at the start of training this is identical to the
    unconditioned DoubleConv."""

    def __init__(self, in_ch, out_ch, embed_dim, device="cpu"):
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
        self.film = FiLM(embed_dim, out_ch)

    def forward(self, x, cond):
        return self.film(self.block(x), cond)


class DownFiLM(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim, device="cpu"):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, padding_mode="circular", device=device),
            nn.GELU(),
        )
        self.conv = DoubleConvFiLM(in_ch, out_ch, embed_dim, device=device)

    def forward(self, x, cond):
        return self.conv(self.down(x), cond)


class UpFiLM(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim, device="cpu"):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2, device=device)
        self.conv = DoubleConvFiLM(in_ch, out_ch, embed_dim, device=device)

    def forward(self, x, skip, cond):
        x = self.upconv(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x, cond)


class UNet2D_FiLM_classifier(nn.Module):
    """U-Net (cf. unet_2D.py/unet_2D_classifier.py) with FiLM conditioning on
    Re at EVERY resolution level (inc, each Down, each Up) -- mirrors PFNO2D's
    approach in Parameterized_Neural_Operator (models/pfno_2D.py: FiLM after
    every Fourier layer), ported to the UNet's DoubleConv blocks. The point of
    comparison against unet_2D_concat_classifier.py: does modulating each
    resolution level's activations help beyond just making Re available at
    the input?

    forward(x, re, predict_class=True) -> (pred, class_logits) if
    predict_class, else pred alone (same contract as FNO2D_classifier /
    FNO2D_classifier_concat, required by training/DiffusionModel.py:Diffusion).
    """

    def __init__(self, input_dim, output_dim, depth=3, base_width=32,
                 param_embed_dim=32, param_hidden_dim=64, param_encoder_layers=2,
                 param_log_transform=True, param_mean=0.0, param_std=1.0,
                 class_mlp_layers=None, n_cat=1000, device="cpu"):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.base_width = base_width
        self.n_cat = n_cat

        self.param_encoder = ParamEncoder(
            embed_dim=param_embed_dim, hidden_dim=param_hidden_dim,
            n_layers=param_encoder_layers, log_transform=param_log_transform,
            param_mean=param_mean, param_std=param_std,
        )

        channels = [base_width * (2 ** i) for i in range(depth + 1)]
        bottleneck_ch = channels[depth]

        self.inc = DoubleConvFiLM(input_dim, channels[0], param_embed_dim, device=device)
        self.downs = nn.ModuleList([
            DownFiLM(channels[i], channels[i + 1], param_embed_dim, device=device) for i in range(depth)
        ])
        self.ups = nn.ModuleList([
            UpFiLM(channels[depth - i], channels[depth - i - 1], param_embed_dim, device=device)
            for i in range(depth)
        ])
        self.outc = nn.Conv2d(channels[0], output_dim, kernel_size=1, device=device)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        if class_mlp_layers is None:
            # +1 for the Re scalar concatenated alongside avg+max pooled features
            # (bottleneck_ch, not base_width, matching UNet2D_classifier's own convention)
            class_mlp_layers = [2 * bottleneck_ch + 1, 4 * bottleneck_ch, n_cat]
        self.class_mlp_layers = class_mlp_layers
        self.classifier = FFNN(class_mlp_layers)

    def _normalize_param(self, re):
        # Same normalization as ParamEncoder itself, exposed separately so the
        # classifier head gets an explicit, undiluted Re scalar (not just the
        # implicit signal baked into FiLM-modulated features) -- same
        # rationale as FNO2D_classifier_concat._normalize_param.
        re = re.reshape(-1, 1).float()
        if self.param_encoder.log_transform:
            re = torch.log(re)
        return (re - self.param_encoder.param_mean) / self.param_encoder.param_std

    def forward(self, x, re, predict_class=True):
        re = re.to(x.device)
        cond = self.param_encoder(re)  # (B, embed_dim)

        # (batch, H, W, C) -> (batch, C, H, W)
        x = x.permute(0, 3, 1, 2)

        skips = []
        x = self.inc(x, cond)
        skips.append(x)
        for down in self.downs[:-1]:
            x = down(x, cond)
            skips.append(x)
        x = self.downs[-1](x, cond)  # bottleneck

        if predict_class:
            avg = self.avg_pool(x).flatten(1)
            mx = self.max_pool(x).flatten(1)
            re_scalar = self._normalize_param(re)
            y = torch.cat([avg, mx, re_scalar], dim=-1)
            y = self.classifier(y)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, cond)

        x = self.outc(x)
        x = x.permute(0, 2, 3, 1)

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

    def count_film_parameters(self):
        """Sum of parameters belonging specifically to FiLM layers (inc +
        each down + each up), for reporting how much of the model is
        dedicated to Re-conditioning vs the base UNet backbone."""
        total = sum(p.numel() for p in self.inc.film.parameters())
        for down in self.downs:
            total += sum(p.numel() for p in down.conv.film.parameters())
        for up in self.ups:
            total += sum(p.numel() for p in up.conv.film.parameters())
        return total
