import torch.nn as nn
import torch
import math
from tqdm import tqdm
from scipy.stats import truncnorm
import torch


class Diffusion(nn.Module):
    def __init__(self, model, timesteps, noise_sampling_coeff=None):
        super().__init__()

        self.model = model
        self.timesteps = timesteps
        self.noise_sampling_coeff = noise_sampling_coeff

        betas = self._cosine_variance_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))

    
    def forward(self, x, noise, predict_noise_level=False):

        if self.noise_sampling_coeff:
            ## Draw from [0,1]
            t=torch.tensor(abs(truncnorm(a=-1/self.noise_sampling_coeff, b=1/self.noise_sampling_coeff,
                                            scale=self.noise_sampling_coeff).rvs(size=(x.shape[0],))))
            ## Normalise to full span of timestep range, and convert to int
            t=t*self.timesteps
            t=t.to(torch.int64).to(x.device)
        else:
            t=torch.randint(0,self.timesteps,(x.shape[0],)).to(x.device)

        x_t = self._forward_diffusion(x, t, noise)

        if predict_noise_level:
            pred_noise, pred_noise_level = self.model(x_t, True)
            return pred_noise, x_t, t, pred_noise_level
        else:
            pred_noise=self.model(x_t)
            return pred_noise, x_t, t


    def _forward_diffusion(self, x_0, t, noise=None):
        """ Run forward diffusion process, i.e. add noise to some input images
        x_0:    input tensors to add noise to
        t:      noise level to add. Can be either a tensor with same length x_0, in which case
                each image can be noised differently. Or just pass a scalar, and the same level of noise
                will be added to each image
        noise:  Tensor of random noise. Can be None, in which case we will generate noise here
        
        returns a tensor of the same shape x_0, where each image has been noised """

        ## If t is just an int, create a tensor for the forward process
        if type(t)==int:
            t=t*torch.ones(len(x_0), device=x_0.device, dtype=torch.int64)

        if noise==None:
            noise=torch.randn_like(x_0).to(x_0.device)

        assert x_0.shape==noise.shape
        #q(x_{t}|x_{0})
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise
    

    
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)
        return betas
