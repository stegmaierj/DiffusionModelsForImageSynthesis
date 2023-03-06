# -*- coding: utf-8 -*-
# Code adapted from https://github.com/w86763777/pytorch-ddpm/blob/master/diffusion.py and https://huggingface.co/blog/annotated-diffusion

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#%%
# Definitions

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    

def beta_schedule(timesteps, schedule='linear'):
    
    if schedule=='cosine':
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
    
    if schedule=='linear':
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps)
                
    if schedule=='quadratic':
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
    
    if schedule=='sigmoid':
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    
    return betas.double()


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


#%%
# Training

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, T, schedule='cosine'):
        assert schedule in ['cosine', 'linear', 'quadratic', 'sigmoid'], 'Unknown schedule "{0}"'.format(schedule)
        super().__init__()

        self.T = T

        
        self.register_buffer('betas', beta_schedule(T, schedule=schedule))
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, t=None):
        """
        Algorithm 1.
        """
        if t==None:
            t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        else:
            t = x_0.new_ones([x_0.shape[0], ], dtype=torch.long) * t
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        return x_t, noise, t

#%%
# Testing

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, T, t_start=1000, t_save=100, t_step=1,\
                 schedule='linear', mean_type='epsilon', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon'], 'Unknown mean_type "{0}"'.format(mean_type)
        assert var_type in ['fixedlarge', 'fixedsmall'], 'Unknown var_type "{0}"'.format(var_type)
        assert schedule in ['cosine', 'linear', 'quadratic', 'sigmoid'], 'Unknown schedule "{0}"'.format(schedule)
        super().__init__()

        self.model = model
        self.T = T
        self.t_start = t_start
        self.t_save = t_save
        self.t_step = t_step
        self.schedule = schedule
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer('betas', beta_schedule(T, schedule=self.schedule))
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, cond, t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(torch.cat((x_t,cond), axis=1), t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(torch.cat((x_t,cond), axis=1), t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(torch.cat((x_t,cond), axis=1), t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var
    
    def __len__(self):
        iterations = torch.linspace(0,self.t_start-1,self.t_start//self.t_step, dtype=int) % self.t_save
        return torch.count_nonzero(iterations==0)

    def forward(self, x_T, cond):
        """
        Algorithm 2.
        """
        x_t = x_T
        x_intermediate = []
        for time_step in reversed(torch.linspace(0,self.t_start-1,self.t_start//self.t_step, dtype=int)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, cond=cond, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
            
            if time_step%self.t_save == 0:
                x_intermediate.append(x_t.cpu())
                print('saved {0}'.format(time_step))
        x_0 = x_t
        return x_0, x_intermediate


