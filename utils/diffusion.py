# -*- coding: utf-8 -*-
# Code adapted from https://github.com/w86763777/pytorch-ddpm/blob/master/diffusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F


#%%
# Definitions


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


#%%
'''
# code adapted from https://github.com/NVlabs/denoising-diffusion-gan/blob/6818ded6443e10ab69c7864745457ce391d4d883/train_ddgan.py#L100

import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from skimage import io



#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.diffusion_schedule=='geometric':
        var = var_func_geometric(t, beta_min, beta_max)
    elif args.diffusion_schedule=='vp':
        var = var_func_vp(t, beta_min, beta_max)
    else:
        raise NotImplementedError('Diffusion schedule "{0}" not implemented.'.format(args.diffusion_schedule))
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas


class Diffusion_Coefficients():
    def __init__(self, args, device):
                
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)
    
    
def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)
      
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    
    return x_t


def q_sample_pairs(coeff, x_start, t, *, noise=None):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    if noise is None:
      noise = torch.randn_like(x_start)
    
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    
    return x_t, x_t_plus_one



#%% posterior sampling DiffusionModels

class Sampling_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        # define alphas 
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


@torch.no_grad()
def sample_from_DiffusionModel(coefficients, model, x_init, cond, c_time, n_time=10, s_time=1):
    
    @torch.no_grad()
    def p_sample(model, x, cond, t, t_index):
        betas_t = extract(coefficients.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            coefficients.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(coefficients.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(torch.cat((x, cond), axis=1), t) / sqrt_one_minus_alphas_cumprod_t
        )
    
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(coefficients.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
    
    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(model, x, cond, c_time, n_time=10, s_time=1):
        device = next(model.parameters()).device
        
        x_intermediate = []
        
        if n_time == -1: n_time = c_time
        else: n_time = np.minimum(n_time, c_time)
        
        for i in tqdm(reversed(range(c_time-n_time, c_time)), desc='sampling loop time step', total=n_time):
            x = p_sample(model, x, cond, torch.full((x.shape[0],), i, device=device, dtype=torch.long), i)
            if i%s_time==0 or i==0:
                x_intermediate.append(x.cpu())
        return x_intermediate

    return p_sample_loop(model, x_init, cond, c_time, n_time=n_time, s_time=s_time)




#%% posterior sampling DiffusionGAN

class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
        
def sample_posterior(coefficients, x_0, x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos

@torch.no_grad()
def sample_from_DiffusionGAN(coefficients, model, x_init, cond, c_time, n_time=10, s_time=1):
    
    x = x_init.clone()
    x_intermediate = []
    
    if n_time == -1: n_time = c_time
    else: n_time = np.minimum(n_time, c_time)
    
    with torch.no_grad():
        for i in tqdm(reversed(range(c_time-n_time, c_time)), desc='sampling loop time step', total=n_time):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            
            t_time = t
            x_0 = model(torch.cat((x, cond), axis=1), t_time)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
            
            if i%s_time==0 or i==0:
                x_intermediate.append(x.cpu())
            
    return x_intermediate



#%%

# from argparse import Namespace
# args = Namespace(**{'beta_min':0.1,'beta_max':5, 'num_timesteps':10, 'diffusion_schedule':'vp'})

def test_args(img_path, args, save_path):
    
    os.makedirs(save_path, exist_ok=True)
    
    img = io.imread(img_path)
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    img -= 0.5*(img_max+img_min)
    img /= 0.5*(img_max-img_min)
    img = torch.from_numpy(img)[None,None,...]
    
    coeffs = Diffusion_Coefficients(args, 'cpu')
    
    for t in range(args.num_timesteps+1):
        if t==0:
            out = q_sample(coeffs, img, torch.tensor([t,]))
        else:
            noise = torch.randn_like(out)
            out = extract(coeffs.a_s, torch.tensor([t,]), out.shape) *out + \
                  extract(coeffs.sigmas, torch.tensor([t,]), out.shape) * noise
        io.imsave(os.path.join(save_path, 't{0:03d}.tif'.format(t)), out.data.numpy()[0,0,0,...])
        

'''