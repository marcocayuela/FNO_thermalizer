import torch
import torch.nn as nn

from fno.fno_2D import FFNN
from unet.unet_2D import DoubleConv, Down, Up


class UNet2D_classifier(nn.Module):
    """U-Net avec tete classifieur branchee sur le bottleneck, pour le modele de
    diffusion (meme role que FNO2D_classifier : predit le bruit ajoute ET, optionnellement,
    le niveau de bruit/timestep, pour la correction de trajectoire du thermalizer).

    forward(x, predict_class=True) -> (pred, class_logits) si predict_class, sinon pred seul
    (meme contrat que FNO2D_classifier, requis par training/DiffusionModel.py:Diffusion).
    """

    def __init__(self, input_dim, output_dim, depth=3, base_width=32,
                 class_mlp_layers=None, n_cat=1000, device="cpu"):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.base_width = base_width
        self.n_cat = n_cat

        channels = [base_width * (2 ** i) for i in range(depth + 1)]
        bottleneck_ch = channels[depth]

        self.inc = DoubleConv(input_dim, channels[0], device=device)
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
            class_mlp_layers = [2 * bottleneck_ch, 4 * bottleneck_ch, n_cat]
        self.classifier = FFNN(class_mlp_layers)

    def forward(self, x, predict_class=True):
        x = x.permute(0, 3, 1, 2)

        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs[:-1]:
            x = down(x)
            skips.append(x)
        bottleneck = self.downs[-1](x)

        if predict_class:
            avg = self.avg_pool(bottleneck).squeeze()
            mx = self.max_pool(bottleneck).squeeze()
            y = torch.cat([avg, mx], dim=-1)
            y = self.classifier(y)

        x = bottleneck
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
