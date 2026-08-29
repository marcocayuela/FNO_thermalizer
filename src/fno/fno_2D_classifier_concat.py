import torch
import torch.nn as nn
import torch.nn.functional as F

from fno.fno_2D import *


class FNO2D_classifier_concat(nn.Module):
    """FNO2D_classifier (cf. fno_2D_classifier.py) with the Reynolds number
    concatenated as an extra input channel -- lets one diffusion corrector be
    trained across several Re instead of one model per Re.

    Re is also concatenated to the classifier head's pooled features (before
    self.classifier): the head estimates the apparent noise level of x_t, and
    "on-manifold for this noise level" only means something once it knows
    which Re's manifold it's comparing against.

    Same normalization convention as PFNO2D/FNO2DConcat in
    Parameterized_Neural_Operator/models/{film,fno_concat_2D}.py:
    log-transform + standardize, stats passed in (computed on re_values).
    """

    def __init__(self, input_dim, output_dim, modes_x, modes_y, width, l, n_layer=4,
                 hidden_proj=None, mlp=True, layers_mlp=None, class_mlp_layers=None, n_cat=1000,
                 param_log_transform=True, param_mean=0.0, param_std=1.0, device="cpu"):
        super(FNO2D_classifier_concat, self).__init__()

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

        if not hidden_proj:
            self.hidden_proj = self.width
        else:
            self.hidden_proj = hidden_proj

        self.padding = 0

        self.activation = nn.LeakyReLU()

        self.param_log_transform = param_log_transform
        self.register_buffer("param_mean", torch.tensor(float(param_mean)))
        self.register_buffer("param_std", torch.tensor(float(param_std)))

        # +2 for the coordinate meshgrid (get_meshgrid), +1 for the concatenated Re channel
        self.P = nn.Linear(self.input_dim + 2 + 1, self.width)

        self.layers = nn.ModuleList()
        for i in range(self.n_layer):
            self.layers.append(FourierLayer2D(self.width, self.width, self.modes_x, self.modes_y, l,
                                              mlp=self.mlp, layers_mlp=self.layers_mlp, device=self.device))

        # classifier
        self.n_cat = n_cat
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        if class_mlp_layers is None:
            # +1 for the Re scalar concatenated alongside avg+max pooled features
            self.class_mlp_layers = [2 * self.width + 1, 4 * self.width, self.n_cat]
        else:
            self.class_mlp_layers = class_mlp_layers

        self.classifier = FFNN(self.class_mlp_layers)

        self.Q = nn.Sequential(nn.Linear(self.width, self.hidden_proj), self.activation,
                               nn.Linear(self.hidden_proj, self.output_dim))

    def _normalize_param(self, re):
        # re: (B,) raw values -> (B, 1) normalized
        re = re.reshape(-1, 1).float()
        if self.param_log_transform:
            re = torch.log(re)
        return (re - self.param_mean) / self.param_std

    def forward(self, x, re, predict_class=True):
        # x: (batch_size, x_size, y_size, input_dim), re: (batch_size,)
        param_norm = self._normalize_param(re.to(x.device))  # (B, 1)

        meshgrid = get_meshgrid(x.shape, self.device)
        B, H, W, _ = x.shape
        param_channel = param_norm.reshape(B, 1, 1, 1).expand(B, H, W, 1)
        x = torch.cat((x, meshgrid, param_channel), dim=-1)

        x = self.P(x)
        x = x.permute(0, 3, 1, 2)

        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for layer in self.layers:
            x = self.activation(layer(x))

        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]

        if predict_class:
            # flatten(1), not squeeze(): squeeze would also collapse the batch
            # dim at batch_size==1 (the only batch size correction_eval.py's
            # rollout ever uses), breaking the cat with param_norm below.
            x_avg_pool = self.avg_pool(x).flatten(1)
            x_max_pool = self.max_pool(x).flatten(1)
            y = torch.cat([x_avg_pool, x_max_pool, param_norm], dim=-1)
            y = self.classifier(y)

        x = x.permute(0, 2, 3, 1)
        x = self.Q(x).squeeze(-1)
        if predict_class:
            return x, y
        else:
            return x

    def count_parameters_per_module(self):
        param_dict = {}
        for name, module in self.named_children():
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            param_dict[name] = trainable
        param_dict["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return param_dict

    def class_distribution(self, x, re):
        meshgrid = get_meshgrid(x.shape, self.device)
        param_norm = self._normalize_param(re.to(x.device))
        B, H, W, _ = x.shape
        param_channel = param_norm.reshape(B, 1, 1, 1).expand(B, H, W, 1)
        x = torch.cat((x, meshgrid, param_channel), dim=-1)
        x = self.P(x)
        x = x.permute(0, 3, 1, 2)

        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for layer in self.layers:
            x = self.activation(layer(x))

        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]

        x_avg_pool = self.avg_pool(x).flatten(1)
        x_max_pool = self.max_pool(x).flatten(1)
        y = torch.cat([x_avg_pool, x_max_pool, param_norm], dim=-1)
        y = self.classifier(y)
        return nn.Softmax(-1)(y)

    def classifier_cat(self, x, re):
        y = self.class_distribution(x, re)
        return y.argmax(-1)
