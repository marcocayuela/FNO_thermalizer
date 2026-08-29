import torch
import torch.nn as nn
import torch.nn.functional as F



def get_meshgrid(shape, device="cpu"):
    batch_size, x_size, y_size, _ = shape

    #batch_size, x_size, y_size = shape[0], shape[1], shape[2]
    x_grid = torch.linspace(0, 1, x_size, device=device)
    x_grid = x_grid.reshape(1, x_size, 1, 1).repeat([batch_size, 1, y_size, 1])

    y_grid = torch.linspace(0, 1, y_size, device=device)
    y_grid = y_grid.reshape(1, 1, y_size, 1).repeat([batch_size, x_size, 1, 1])

    grid = torch.concat((x_grid, y_grid), dim=-1).float()
    return grid



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
    

class FFNN2D(nn.Module):
    """
    Feed Forward Neural Network pointwise 2D avec Conv2d kernel_size=1.

    Args:
        layers_width (list): liste des tailles de chaque couche (channels) [A, B, C, ...]
        activation (nn.Module, optional): fonction d'activation entre les couches. Defaults to nn.GELU().
        device (str, optional): device. Defaults to "cpu".
    """

    def __init__(self, layers_width, activation=nn.GELU(), device="cpu"):
        super(FFNN2D, self).__init__()

        self.layers_width = layers_width
        self.depth = len(layers_width)
        self.activation = activation

        layers = []
        for i in range(self.depth - 2):
            layers.append(nn.Conv2d(layers_width[i], layers_width[i+1], kernel_size=1, device=device))
            layers.append(self.activation)
        # dernière couche sans activation
        layers.append(nn.Conv2d(layers_width[-2], layers_width[-1], kernel_size=1, device=device))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: shape (batch, channels, x_dim, y_dim)
        returns: shape (batch, channels_out, x_dim, y_dim)
        """
        return self.model(x)
    



class IntegralKernel2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super(IntegralKernel2D, self).__init__()
        """
        2D Fourier layer. It computes FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels  # d_v
        self.out_channels = out_channels  # d_v
        self.modes_x = modes_x  # k_max on x 
        self.modes_y = modes_y  # k_max on y 
        
        self.scale = (1 / (in_channels * out_channels))

        # Parametrization R of the kernel
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 2*self.modes_x, self.modes_y, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]  # Batch size
        dv = x.shape[1]  # Lifting-dimension
        lastdim = x.shape[-1]  # Size of last dimension

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)
        del x 
        x_ft = torch.fft.fftshift(x_ft, dim=-2)

        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-2], x_ft.shape[-1], device=x_ft.device, dtype=torch.cfloat)
        midX =  x_ft.shape[-2] // 2

        # Use compl_mul1d to perform the multiplication between the relevant Fourier Modes and self.weights and fill the tensor out_tf with the corresponding values
        out_ft[:, :, midX-self.modes_x:midX+self.modes_x, :self.modes_y] = self.compl_mul2d(x_ft[:, :, midX-self.modes_x:midX+self.modes_x, :self.modes_y], self.weights)
        
        del x_ft
        # Return to physical space
        out_ft = torch.fft.fftshift(out_ft, dim=-2)
        x_out = torch.fft.irfft2(out_ft, s=(out_ft.shape[-2], lastdim))
        return x_out

    


class FourierLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y, l=1, mlp=True, layers_mlp=None, device="cpu"):
        super(FourierLayer2D, self).__init__()

        self.int_kern = IntegralKernel2D(in_channels, out_channels, modes_x, modes_y)
        self.w = nn.Conv2d(in_channels, out_channels, l, padding='same')
        if layers_mlp is not None:
            self.mlp = FFNN2D(layers_mlp, device=device) if mlp else None
        else:
            self.mlp = FFNN2D(3*[out_channels], device=device)

    def forward(self, x):

        if self.mlp:
            return self.mlp(self.int_kern(x)) + self.w(x)
        else:
            return self.int_kern(x) + self.w(x)



class FNO2D(nn.Module):
    def __init__(self, input_dim, output_dim, modes_x, modes_y, width, l, n_layer=4, hidden_proj=None, mlp=True, layers_mlp=None, padding=0, device="cpu"):
        super(FNO2D, self).__init__()

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

        # Domain padding before the FFT (Li et al. 2021): weakens FNO's
        # implicit periodicity assumption for a non-periodic domain (e.g.
        # euler_multi_quadrants_openBC's open BC). Was hardcoded to 0 here
        # (dead code) -- exposed as a constructor arg instead, mirroring
        # RT_FNO_vs_WNO/src/models/fno_2D.py's identical fix. Default 0 =
        # unchanged behavior for existing (periodic) Kolmogorov/KS configs.
        self.padding = padding

        self.activation = nn.LeakyReLU()

        # Lifting transformation, corresponding to a simple linear transformation
        self.P = nn.Linear(self.input_dim + 2, self.width)  # input channel is 4: (u(x,y),v(x,y), x, y)
        
        
        # n_l sequential layers
        self.layers = nn.ModuleList()
        for i in range(self.n_layer):
            self.layers.append(FourierLayer2D(self.width, self.width, self.modes_x, self.modes_y, l, mlp=self.mlp, layers_mlp=self.layers_mlp, device=self.device))


        # Projection layer
        self.Q = nn.Sequential(nn.Linear(self.width, self.hidden_proj), self.activation, nn.Linear(self.hidden_proj, self.output_dim))
        #self.Q = nn.Sequential(nn.Linear(self.width, self.width), self.activation, nn.Linear(self.width, 1))


    def forward(self, x):
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

        # Comeback to the origin position of the axes
        # (batch_size, lifting_dimension, x_size, y_size) -> (batch_size, x_size, y_size, lifting_dimension)
        x = x.permute(0, 2, 3, 1)
     
        # Apply projection to go back to the original space (non-lifted)

        x = self.Q(x).squeeze(-1)
        return x
    
    def count_parameters_per_module(self):

        param_dict = {}
        
        for name, module in self.named_children():
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            param_dict[name] = trainable
        
        param_dict["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return param_dict