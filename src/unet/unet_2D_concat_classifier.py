import torch
import torch.nn as nn

from unet.unet_2D import DoubleConv, Down, Up
from fno.fno_2D import FFNN


class UNet2D_concat_classifier(nn.Module):
    """U-Net (cf. unet_2D.py/unet_2D_classifier.py) with Re concatenated as an
    extra input channel -- the "naive conditioning" counterpart to
    unet_2D_film_classifier.py's per-level FiLM. No meshgrid coordinate
    channel here (unlike FNO2D_classifier_concat): UNet's convolutions are
    local/translation-equivariant, they don't need explicit x/y coordinates
    the way FNO's global spectral mixing does -- so it's input_dim + 1, not
    input_dim + 2 + 1.

    forward(x, re, predict_class=True) -> (pred, class_logits) if
    predict_class, else pred alone (same contract as the other diffusion
    backbones, required by training/DiffusionModel.py:Diffusion).

    Numerical note (confirmed empirically, robust across 5 seeds): the Re
    channel's effect on the output collapses to numerical noise (diff/std ~
    1e-6) when base_width <= 8, because GroupNorm's n_groups = min(8, out_ch)
    then equals out_ch -- normalization degenerates to one group per
    channel, and since the Re channel's contribution to each output channel
    is spatially CONSTANT (broadcast, same at every pixel), per-channel
    spatial-mean subtraction removes nearly all of it. At base_width=32
    (n_groups=8 < out_ch), this doesn't happen -- Re-sensitivity is strong
    and consistent (diff/std ~ 3-5 across seeds). Only matters for small
    smoke-test configs; production width (32) is unaffected.
    """

    def __init__(self, input_dim, output_dim, depth=3, base_width=32,
                 param_log_transform=True, param_mean=0.0, param_std=1.0,
                 class_mlp_layers=None, n_cat=1000, device="cpu"):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.base_width = base_width
        self.n_cat = n_cat

        self.param_log_transform = param_log_transform
        self.register_buffer("param_mean", torch.tensor(float(param_mean)))
        self.register_buffer("param_std", torch.tensor(float(param_std)))

        channels = [base_width * (2 ** i) for i in range(depth + 1)]
        bottleneck_ch = channels[depth]

        self.inc = DoubleConv(input_dim + 1, channels[0], device=device)
        self.downs = nn.ModuleList([
            Down(channels[i], channels[i + 1], device=device) for i in range(depth)
        ])
        self.ups = nn.ModuleList([
            Up(channels[depth - i], channels[depth - i - 1], device=device) for i in range(depth)
        ])
        self.outc = nn.Conv2d(channels[0], output_dim, kernel_size=1, device=device)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        if class_mlp_layers is None:
            class_mlp_layers = [2 * bottleneck_ch + 1, 4 * bottleneck_ch, n_cat]
        self.class_mlp_layers = class_mlp_layers
        self.classifier = FFNN(class_mlp_layers)

    def _normalize_param(self, re):
        re = re.reshape(-1, 1).float()
        if self.param_log_transform:
            re = torch.log(re)
        return (re - self.param_mean) / self.param_std

    def forward(self, x, re, predict_class=True):
        re = re.to(x.device)
        re_norm = self._normalize_param(re)  # (B, 1)

        # (batch, H, W, C) -> (batch, C, H, W), concat Re as an extra channel
        x = x.permute(0, 3, 1, 2)
        B, _, H, W = x.shape
        re_channel = re_norm.reshape(B, 1, 1, 1).expand(B, 1, H, W)
        x = torch.cat([x, re_channel], dim=1)

        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs[:-1]:
            x = down(x)
            skips.append(x)
        x = self.downs[-1](x)  # bottleneck

        if predict_class:
            avg = self.avg_pool(x).flatten(1)
            mx = self.max_pool(x).flatten(1)
            y = torch.cat([avg, mx, re_norm], dim=-1)
            y = self.classifier(y)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

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
