import torch
import torch.nn as nn
import torch.nn.functional as F

from fno.fno_2D import *




class FNO2D_classifier(nn.Module):
    def __init__(self, input_dim, output_dim, modes_x, modes_y, width, l, n_layer=4, hidden_proj=None, mlp=True, layers_mlp=None, class_mlp_layers=None, n_cat=1000, padding=0, device="cpu"):
        super(FNO2D_classifier, self).__init__()

        self.device = device

        self.modes_x = modes_x  # k_modes on x
        self.modes_y = modes_y  # k_modes on y
        self.width = width  # d_v
        self.l = l  #kernel size in linear transformation
        self.n_layer = n_layer #nbr of layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = mlp
        self.layers_mlp = layers_mlp

        if not hidden_proj:
            self.hidden_proj = self.width
        else:
            self.hidden_proj = hidden_proj

        # Domain padding before the FFT -- see fno_2D.py::FNO2D's identical
        # fix/comment. Default 0 = unchanged behavior for existing configs.
        self.padding = padding

        self.activation = nn.LeakyReLU()

        # Lifting transformation, corresponding to a simple linear transformation
        self.P = nn.Linear(self.input_dim + 2, self.width)  # input channel is 4: (u(x,y),v(x,y), x, y)
        
        
        # n_l sequential layers
        self.layers = nn.ModuleList()
        for i in range(self.n_layer):
            self.layers.append(FourierLayer2D(self.width, self.width, self.modes_x, self.modes_y, l, mlp=self.mlp, layers_mlp=self.layers_mlp, device=self.device))

        # classifier
        self.n_cat = n_cat
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        if class_mlp_layers is None:
            self.class_mlp_layers = [2*self.width, 4*self.width, self.n_cat]
        else:
            self.class_mlp_layers = class_mlp_layers

        self.classifier = FFNN(class_mlp_layers)

        # Projection layer
        self.Q = nn.Sequential(nn.Linear(self.width, self.hidden_proj), self.activation, nn.Linear(self.hidden_proj, self.output_dim))
        #self.Q = nn.Sequential(nn.Linear(self.width, self.width), self.activation, nn.Linear(self.width, 1))


    def forward(self, x, predict_class=True):
        # input_grid must be the concatenation of X and Y (meshgrids) shape must be (x_size, y_size, 2)
        # shape of x is (batch_size, x_size, y_size)
        
        meshgrid = get_meshgrid(x.shape, self.device)
        # Concatenate the grid to the input to gt the term v0
        x = torch.cat((x, meshgrid), dim=-1)

        # Apply the lifting transformation
        x = self.P(x)

        # Permute the axis, so that the axis corresponding to the physical space is the last (in order to compute the FFT)
        x = x.permute(0, 3, 1, 2)

        # Add padding (if input is non-periodic)
        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        # Apply n_l integral layers
        for layer in self.layers:
            x = self.activation(layer(x))

        # Remove padding if input is non-periodic
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]

        if predict_class:
            # we do not apply softmax because cross-entropy loss already does
            # Squeeze only the trailing H=1,W=1 spatial dims (bare .squeeze()
            # also collapsed the batch dim whenever batch_size==1, turning
            # (1,width) into (width,) and crashing cross_entropy downstream
            # with a shape mismatch -- hit for real on a 25-sample dataset
            # with batch_size=4, whose last batch has exactly 1 sample).
            x_avg_pool = self.avg_pool(x).squeeze(-1).squeeze(-1)
            x_max_pool = self.max_pool(x).squeeze(-1).squeeze(-1)
            y = torch.cat([x_avg_pool, x_max_pool], dim=-1)
            y = self.classifier(y)

        # Comeback to the origin position of the axes
        # (batch_size, lifting_dimension, x_size, y_size) -> (batch_size, x_size, y_size, lifting_dimension)
        x = x.permute(0, 2, 3, 1)
     
        # Apply projection to go back to the original space (non-lifted)

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
    
    def class_distribution(self, x: torch.Tensor):
        """ Forward pass with only a regressor output"""
    

        meshgrid = get_meshgrid(x.shape, self.device)
        x = torch.cat((x, meshgrid), dim=-1)
        x = self.P(x)
        x = x.permute(0, 3, 1, 2)

        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for layer in self.layers:
            x = self.activation(layer(x))

        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]

        x = self.classifier(x)
        x = nn.Softmax(-1)(x)
        return x
    
    def classifier_cat(self, x: torch.Tensor):
        x = self.class_distribution(x)
        return x.argmax(-1)
    
