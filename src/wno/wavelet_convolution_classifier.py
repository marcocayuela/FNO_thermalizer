import torch
import torch.nn as nn
import torch.nn.functional as F
from .wavelet_convolution import WaveConv2d, WaveConv2dCwt



class FFNN(nn.Module):

    """Feed Forward Neural Network 
    Args:
        layers_width (list): list of integers representing the width of each layer;
        activation (nn.Module, optional): activation function to use between layers. Defaults to nn.GELU();
        device (str, optional): device to run the model on. Defaults to "cpu".
    Returns:
        nn.Module: Feed Forward Neural Network model.
    """   

    def __init__(self, layers_width, activation=nn.GELU(), device="cpu"):
        super(FFNN, self).__init__()

        self.layers_width = layers_width
        self.depth = len(self.layers_width)
        self.activation = activation

        self.model = nn.Sequential()
        for i in range(self.depth - 2):
            self.model.append(nn.Linear(self.layers_width[i], self.layers_width[i+1], device=device))
            self.model.append(self.activation)
        self.model.append(nn.Linear(self.layers_width[-2], self.layers_width[-1], device=device))

    def forward(self, input):
        return self.model(input)
    


class WNO2d_classifier(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, mode, in_channel, out_channel, grid_range, padding=0, class_mlp_layers=None, n_cat=100, device="cpu"):
        super(WNO2d_classifier, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : (T_in+1)-channel tensor, solution at t0-t_T and location (u(x,y,t0),...u(x,y,t_T), x,y)
              : shape: (batchsize * x=width * x=height * c=T_in+1)
        Output: Solution of a later timestep (u(x, T_in+1))
              : shape: (batchsize * x=width * x=height * c=1)
        
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : list with 2 elements (for 2D), image size
        wavelet   : list of strings, first and second level continuous wavelet filters
        in_channel: scalar, channels in input including grid
        grid_range: list with 2 elements (for 2D), right supports of 2D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.mode = mode
        if isinstance(self.wavelet, list):
            self.wavelet1 = self.wavelet[0]
            self.wavelet2 = self.wavelet[1]
        else:
            self.wavelet = wavelet
        self.device = device
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.grid_range = grid_range 
        self.padding = padding
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        self.fc0 = nn.Linear(self.in_channel + 2, self.width, device=self.device) 
        for i in range( self.layers ):
            if isinstance(self.wavelet, list):
                self.conv.append(WaveConv2dCwt(self.width, self.width, self.level, self.size,
                                               self.wavelet1, self.wavelet2, device=self.device))
            else:
                self.conv.append(WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet, device=self.device, mode=self.mode))
            self.w.append(nn.Conv2d(self.width, self.width, 1, device=self.device))
        self.fc1 = nn.Linear(self.width, 128, device=self.device)
        self.fc2 = nn.Linear(128, self.out_channel, device=self.device)

        # classifier
        self.n_cat = n_cat
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        if class_mlp_layers is None:
            self.class_mlp_layers = [2*self.width, 4*self.width, self.n_cat]
        else:
            self.class_mlp_layers = class_mlp_layers

        self.classifier = FFNN(class_mlp_layers)

    def forward(self, x, predict_class=True):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)    
        x = self.fc0(x)                      # Shape: Batch * x * y * Channel
        x = x.permute(0, 3, 1, 2)            # Shape: Batch * Channel * x * y
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding]) 
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.gelu(x)                # Shape: Batch * Channel * x * y
                
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding] 

        if predict_class:
            # we do not apply softmax because cross-entropy loss already does
            x_avg_pool = self.avg_pool(x).squeeze()
            x_max_pool = self.max_pool(x).squeeze()
            y = torch.cat([x_avg_pool, x_max_pool], dim=-1)
            y = self.classifier(y)

        x = x.permute(0, 2, 3, 1)            # Shape: Batch * x * y * Channel
        x = F.gelu( self.fc1(x) )            # Shape: Batch * x * y * Channel
        x = self.fc2(x)                      # Shape: Batch * x * y * Channel
        if predict_class:
            return x, y
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, self.grid_range[0], size_x, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, self.grid_range[1], size_y, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

    def count_parameters_per_module(self):

        param_dict = {}
        
        for name, module in self.named_children():
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            param_dict[name] = trainable
        
        param_dict["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return param_dict
    
    def class_distribution(self, x: torch.Tensor):
        """ Forward pass with only a regressor output"""
    

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)    
        x = self.fc0(x)                      # Shape: Batch * x * y * Channel
        x = x.permute(0, 3, 1, 2)            # Shape: Batch * Channel * x * y
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding])

        for index, (convl, wl) in enumerate( zip(self.conv, self.w)):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.gelu(x)   

        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]

        x = self.classifier(x)
        x = nn.Softmax(-1)(x)
        return x
    
    def classifier_cat(self, x: torch.Tensor):
        x = self.class_distribution(x)
        return x.argmax(-1)