import torch.nn as nn
import torch
import math
from tqdm import tqdm
from scipy.stats import truncnorm
import torch


class Diffusion(nn.Module):
    def __init__(self, model, timesteps, noise_sampling_coeff=None, x_mean=0.0, x_std=1.0):
        super().__init__()

        self.model = model
        self.timesteps = timesteps
        self.noise_sampling_coeff = noise_sampling_coeff

        betas = self._cosine_variance_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas",betas.to(self.model.device))
        self.register_buffer("alphas",alphas.to(self.model.device))
        self.register_buffer("alphas_cumprod",alphas_cumprod.to(self.model.device))
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod).to(self.model.device))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod).to(self.model.device))

        # Field (u,v) normalization -- identity by default (x_mean=0, x_std=1),
        # so existing checkpoints/configs are unaffected. Registered as
        # buffers (same mechanism as ParamEncoder's param_mean/param_std) so
        # they persist in the checkpoint. The DDPM noise schedule above
        # implicitly assumes x_0 has ~unit variance (at t=0, alphas_cumprod~=1
        # so x_t~=x_0; as t->T, x_t->N(0,I) regardless of x_0's own scale) --
        # normalizing keeps that assumption valid for raw (u,v) fields whose
        # variance isn't 1.
        # Always shaped (output_dim,), even for the scalar 0.0/1.0 defaults:
        # load_diffusion()/load_diffusion_concat() construct a fresh Diffusion
        # with these defaults and then overwrite every buffer via
        # load_state_dict(checkpoint) (same placeholder pattern already used
        # for param_mean/param_std) -- a shape mismatch there (e.g. scalar ()
        # vs a normalized checkpoint's (2,)) would make that load_state_dict
        # call fail, so the shape must not depend on whether x_mean/x_std
        # happen to be scalar or per-channel. A bare (output_dim,) buffer
        # (not reshaped to (1,1,1,-1)) broadcasts correctly against the last
        # dim regardless of the field's rank -- (B,H,W,C) for 2D Kolmogorov
        # fields, (B,Nx,C) for a 1D KS field alike.
        def _as_tensor(v):
            t = torch.as_tensor(v, dtype=torch.float32).reshape(-1)
            return t.expand(self.model.output_dim).clone()
        self.register_buffer("x_mean", _as_tensor(x_mean).to(self.model.device))
        self.register_buffer("x_std", _as_tensor(x_std).to(self.model.device))

    def normalize(self, x):
        return (x - self.x_mean) / self.x_std

    def denormalize(self, x):
        return x * self.x_std + self.x_mean

    def forward(self, x, noise, predict_noise_level=False, re=None):
        x = self.normalize(x)

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
            if re is not None:
                pred_noise, pred_noise_level = self.model(x_t, re, True)
            else:
                pred_noise, pred_noise_level = self.model(x_t, True)
            return pred_noise, x_t, t, pred_noise_level
        else:
            pred_noise = self.model(x_t, re) if re is not None else self.model(x_t)
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
        # Per-sample scalar broadcast shape sized to x_0's actual rank (not
        # hardcoded to 4D) -- (B,1,1,1) for a 2D Kolmogorov field (B,H,W,C),
        # (B,1,1) for a 1D KS field (B,Nx,C), any rank in general.
        bshape = (x_0.shape[0],) + (1,) * (x_0.dim() - 1)
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(bshape)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(bshape)*noise

    def sampling(self, n_samples, resolution, device="cpu", re=None):
        """ Generate fresh samples from pure noise """
        x_t=torch.randn((n_samples, resolution, resolution,self.model.output_dim)).to(device)

        for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):
            noise=torch.randn_like(x_t).to(device)
            t=torch.tensor([i for _ in range(n_samples)], dtype=torch.long).to(device)
            x_t=self._reverse_diffusion(x_t,t,noise,re=re)
        return self.denormalize(x_t)


    def _reverse_diffusion(self, x_t, t, noise, re=None):
        '''
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        '''
        pred = self.model(x_t, re, predict_class=False) if re is not None else self.model(x_t, predict_class=False)

        # Per-sample scalar broadcast shape sized to x_t's actual rank (not
        # hardcoded to 4D) -- see the identical fix/comment in _forward_diffusion.
        bshape = (x_t.shape[0],) + (1,) * (x_t.dim() - 1)
        alpha_t=self.alphas.gather(-1,t).reshape(bshape)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(bshape)
        beta_t=self.betas.gather(-1,t).reshape(bshape)
        sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(bshape)
        mean=(1./torch.sqrt(alpha_t))*(x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t)*pred)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(bshape)
            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            std=0.0

        return mean+std*noise 

    
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)
        return betas
