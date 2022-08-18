import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image

import numpy as np
from tqdm import tqdm
from einops import rearrange

import torchgeometry as tgm
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch import linalg as LA

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# helpers functions

import os
import errno

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')

        h = self.net(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_time_emb = True,
        residual = False
    ):
        super().__init__()
        self.channels = channels
        self.residual = residual
        print("Is Time embed used ? ", with_time_emb)

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, time_emb_dim = time_dim, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ConvNextBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        orig_x = x
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)
        if self.residual:
            return self.final_conv(x) + orig_x

        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)

import torch
import torchvision

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        device_of_kernel,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        resolution_routine = 'Incremental',
        train_routine = 'Final',
        sampling_routine='default'
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.device_of_kernel = device_of_kernel

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.resolution_routine = resolution_routine

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.func = self.get_funcs()
        self.train_routine = train_routine
        self.sampling_routine = sampling_routine

    def transform_func(self, img, dec_size, mode, do_blur=False):
        if do_blur:

            kernel = tgm.image.get_gaussian_kernel2d((3, 3), (0.5, 0.5))
            
            conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3,
                             padding=int((3 - 1) / 2), padding_mode='reflect',
                             bias=False, groups=self.channels)
            with torch.no_grad():
                kernel = torch.unsqueeze(kernel, 0)
                kernel = torch.unsqueeze(kernel, 0)
                kernel = kernel.repeat(self.channels, 1, 1, 1).to(img.device)
                conv.weight = nn.Parameter(kernel)

            img = conv(img)
            # blur it

        img_1 = F.interpolate(img, size=img.shape[2] - dec_size, mode=mode, antialias=False)
        img_1 = F.interpolate(img_1, size=img.shape[2], mode='nearest-exact', antialias=False)

        if do_blur:
            kernel = tgm.image.get_gaussian_kernel2d((3, 3), (0.5, 0.5))
            conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3,
                             padding=int((3 - 1) / 2), padding_mode='reflect',
                             bias=False, groups=self.channels)
            with torch.no_grad():
                kernel = torch.unsqueeze(kernel, 0)
                kernel = torch.unsqueeze(kernel, 0)
                kernel = kernel.repeat(self.channels, 1, 1, 1).to(img.device)
                conv.weight = nn.Parameter(kernel)

            img_1 = conv(img_1)

        return img_1

    def get_funcs(self):
        all_funcs = []
        for i in range(self.num_timesteps):
            if self.resolution_routine == 'Incremental':
                all_funcs.append((lambda img, d=i, mode='bicubic': self.transform_func(img, d, mode)))
            elif self.resolution_routine == 'Incremental_bilinear':
                all_funcs.append((lambda img, d=i, mode='bilinear': self.transform_func(img, d, mode)))
            elif self.resolution_routine == 'Incremental_area':
                all_funcs.append((lambda img, d=i, mode='area': self.transform_func(img, d, mode)))

            elif self.resolution_routine == 'Incremental_bicubic_with_blur':
                all_funcs.append((lambda img, d=i, mode='bicubic', do_blur=True: self.transform_func(img, d, mode, do_blur)))
            elif self.resolution_routine == 'Incremental_bilinear_with_blur':
                all_funcs.append((lambda img, d=i, mode='bilinear', do_blur=True: self.transform_func(img, d, mode, do_blur)))
            elif self.resolution_routine == 'Incremental_area_with_blur':
                all_funcs.append((lambda img, d=i, mode='area', do_blur=True: self.transform_func(img, d, mode, do_blur)))

            ## factor 2
            elif self.resolution_routine == 'Incremental_factor_2':
                all_funcs.append((lambda img, d=self.image_size -self.image_size // 2**(i+1), mode='bicubic': self.transform_func(img, d, mode)))
            elif self.resolution_routine == 'Incremental_bilinear_factor_2':
                all_funcs.append((lambda img, d=self.image_size -self.image_size // 2**(i+1), mode='bilinear': self.transform_func(img, d, mode)))
            elif self.resolution_routine == 'Incremental_area_factor_2':
                all_funcs.append((lambda img, d=self.image_size -self.image_size // 2**(i+1), mode='area': self.transform_func(img, d, mode)))

        return all_funcs


    @torch.no_grad()
    def sample(self, batch_size = 16, img=None, t=None):

        if t==None:
            t=self.num_timesteps

        for i in range(t):
            with torch.no_grad():
                img = self.func[i](img)

        # 3(2), 2(1), 1(0)
        xt = img
        direct_recons = None
        while(t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x = self.denoise_fn(img, step)

            if self.train_routine == 'Final':
                if direct_recons == None:
                    direct_recons = x

                if self.sampling_routine == 'default':
                    for i in range(t-1):
                        with torch.no_grad():
                            x = self.func[i](x)

                elif self.sampling_routine == 'x0_step_down':
                    x_times = x
                    for i in range(t):
                        with torch.no_grad():
                            x_times = self.func[i](x_times)

                    x_times_sub_1 = x
                    for i in range(t - 1):
                        with torch.no_grad():
                            x_times_sub_1 = self.func[i](x_times_sub_1)

                    x = img - x_times + x_times_sub_1

            img = x
            t = t - 1

        return xt, direct_recons, img

    @torch.no_grad()
    def gen_sample(self, batch_size=16, img=None, t=None, times=None, noise_level=0):
        print("Here?")
        if t == None:
            t = self.num_timesteps
        if times == None:
            times = t
        
        noise = torch.randn_like(img) * noise_level
        img = img + noise

        xt = img 
        direct_recons = None
       
        while (times):
            step = torch.full((batch_size,), times - 1, dtype=torch.long).cuda()
            x = self.denoise_fn(img, step)
                        
            if direct_recons == None:
                direct_recons = x
            
                        
            if self.train_routine == 'Final':
                if self.sampling_routine == 'default':
                    x_times_sub_1 = x
                    for i in range(times-1):
                        with torch.no_grad():
                            x_times_sub_1 = self.func[i](x_times_sub_1)
                    x = x_times_sub_1
                elif self.sampling_routine == 'x0_step_down':
                    print("x0_step_down")
                    x_times = x
                    for i in range(times):
                        with torch.no_grad():
                            x_times = self.func[i](x_times)
                    x_times_sub_1 = x
                    for i in range(times - 1):
                        with torch.no_grad():
                            x_times_sub_1 = self.func[i](x_times_sub_1)
                    x = img - x_times + x_times_sub_1
            img = x
            times = times - 1

        return xt, direct_recons, img

    @torch.no_grad()
    def all_sample(self, batch_size=16, img=None, t=None, times=None):

        print("Here ?")

        if t == None:
            t = self.num_timesteps
        if times == None:
            times = t

        for i in range(t):
            with torch.no_grad():
                img = self.func[i](img)

        X_0s = []
        X_ts = []

        # 3(2), 2(1), 1(0)
        while (times):
            step = torch.full((batch_size,), times - 1, dtype=torch.long).cuda()
            x = self.denoise_fn(img, step)
            X_0s.append(x)
            X_ts.append(img)

            if self.train_routine == 'Final':
                if self.sampling_routine == 'default':
                    x_times_sub_1 = x
                    for i in range(times-1):
                        with torch.no_grad():
                            x_times_sub_1 = self.func[i](x_times_sub_1)

                    x = x_times_sub_1


                elif self.sampling_routine == 'x0_step_down':
                    print("x0_step_down")
                    x_times = x
                    for i in range(times):
                        with torch.no_grad():
                            x_times = self.func[i](x_times)

                    x_times_sub_1 = x
                    for i in range(times - 1):
                        with torch.no_grad():
                            x_times_sub_1 = self.func[i](x_times_sub_1)

                    x = img - x_times + x_times_sub_1


            img = x
            times = times - 1

        return X_0s, X_ts


    @torch.no_grad()
    def forward_and_backward(self, batch_size=16, img=None, t=None, times=None, eval=True):

        if eval:
            self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps
        if times == None:
            times = t

        Forward = []
        Forward.append(img)

        for i in range(t):
            with torch.no_grad():
                img = self.func[i](img)
                Forward.append(img)

        Backward = []
        temp = img

        # 3(2), 2(1), 1(0)
        while (times):
            print(times)
            step = torch.full((batch_size,), times - 1, dtype=torch.long).cuda()
            x = self.denoise_fn(img, step)
            Backward.append(img)

            if self.train_routine == 'Final':
                if self.sampling_routine == 'default':
                    x_times_sub_1 = x
                    for i in range(times-1):
                        with torch.no_grad():
                            x_times_sub_1 = self.func[i](x_times_sub_1)

                    x = x_times_sub_1

                elif self.sampling_routine == 'x0_step_down':
                    print("x0_step_down")
                    x_times = x
                    for i in range(times):
                        with torch.no_grad():
                            x_times = self.func[i](x_times)

                    x_times_sub_1 = x
                    for i in range(times - 1):
                        with torch.no_grad():
                            x_times_sub_1 = self.func[i](x_times_sub_1)

                    x = img - x_times + x_times_sub_1

            img = x
            times = times - 1

        return Forward, Backward, img


    @torch.no_grad()
    def opt(self, img, t=None):
        if t is None:
            t = self.num_timesteps

        for i in range(t):
            with torch.no_grad():
                img = self.func[i](img)

        return img

    def q_sample(self, x_start, t):

        max_iters = torch.max(t)
        all_blurs = []
        x = x_start
        for i in range(max_iters+1):
            with torch.no_grad():
                x = self.func[i](x)
                all_blurs.append(x)

        all_blurs = torch.stack(all_blurs)

        choose_blur = []

        for step in range(t.shape[0]):
            if step != -1:
                choose_blur.append(all_blurs[t[step], step])
            else:
                choose_blur.append(x_start[step])

        choose_blur = torch.stack(choose_blur)

        return choose_blur


    def p_losses(self, x_start, t):
        b, c, h, w = x_start.shape
        if self.train_routine == 'Final':
            x_blur = self.q_sample(x_start=x_start, t=t)
            x_recon = self.denoise_fn(x_blur, t)

            if self.loss_type == 'l1':
                loss = (x_start - x_recon).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()

        elif self.train_routine == 'Final_small_noise':
            x_start = x_start + 0.001*torch.randn_like(x_start)
            x_blur = self.q_sample(x_start=x_start, t=t)
            x_recon = self.denoise_fn(x_blur, t)

            if self.loss_type == 'l1':
                loss = (x_start - x_recon).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()

        elif self.train_routine == 'Final_random_mean':

            mean = torch.mean(x_start, [2, 3])
            new_mean = torch.randn_like(mean)
            new_mean = new_mean.unsqueeze(2).repeat(1, 1, x_start.shape[2])
            new_mean = new_mean.unsqueeze(3).repeat(1, 1, 1, x_start.shape[3])

            mean = torch.mean(x_start, [2, 3], keepdim=True)

            x_start = x_start - mean + new_mean

            x_blur = self.q_sample(x_start=x_start, t=t)
            x_recon = self.denoise_fn(x_blur, t)

            if self.loss_type == 'l1':
                loss = (x_start - x_recon).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()

        elif self.train_routine == 'Final_random_mean_and_actual':

            x_blur = self.q_sample(x_start=x_start, t=t)
            x_recon = self.denoise_fn(x_blur, t)

            if self.loss_type == 'l1':
                loss1 = (x_start - x_recon).abs().mean()
            elif self.loss_type == 'l2':
                loss1 = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()

            mean = torch.mean(x_start, [2, 3])
            new_mean = torch.randn_like(mean)
            new_mean = new_mean.unsqueeze(2).repeat(1, 1, x_start.shape[2])
            new_mean = new_mean.unsqueeze(3).repeat(1, 1, 1, x_start.shape[3])

            mean = torch.mean(x_start, [2, 3], keepdim=True)
            x_start = x_start - mean + new_mean

            x_blur = self.q_sample(x_start=x_start, t=t)
            x_recon = self.denoise_fn(x_blur, t)

            if self.loss_type == 'l1':
                loss2 = (x_start - x_recon).abs().mean()
            elif self.loss_type == 'l2':
                loss2 = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()

            loss = loss1 + loss2


        elif self.train_routine == 'Gradient_norm':
            x_blur = self.q_sample(x_start=x_start, t=t)
            grad_pred = self.denoise_fn(x_blur, t)
            gradient = (x_blur - x_start)
            norm = LA.norm(gradient, dim=(1,2,3), keepdim=True)
            gradient_norm = gradient/(norm + 1e-5)

            if self.loss_type == 'l1':
                loss = (gradient_norm - grad_pred).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(gradient_norm, grad_pred)
            else:
                raise NotImplementedError()

        elif self.train_routine == 'Step':
            x_blur = self.q_sample(x_start=x_start, t=t)
            x_blur_sub = self.q_sample(x_start=x_start, t=t-1)

            x_blur_sub_pred = self.denoise_fn(x_blur, t)

            if self.loss_type == 'l1':
                loss = (x_blur_sub - x_blur_sub_pred).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_blur_sub, x_blur_sub_pred)
            else:
                raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)

# dataset classes

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


class Dataset_Aug1(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

class Dataset_Aug2(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        #save_and_sample_every = 10,
        results_folder = './results',
        load_path = None,
        dataset = None,
        shuffle=True
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.module.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        if dataset == 'cifar10' or dataset == 'celebA':
            print(dataset, "DA used")
            self.ds = Dataset_Aug1(folder, image_size)
        elif dataset == 'flower':
            print(dataset)
            self.ds = Dataset_Aug2(folder, image_size)
        else:
            self.ds = Dataset(folder, image_size)

        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=shuffle, pin_memory=True, num_workers=1))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.reset_parameters()

        if load_path != None:
            self.load(load_path)


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])


    def train(self):

        backwards = partial(loss_backwards, self.fp16)

        acc_loss = 0
        while self.step < self.train_num_steps:
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()
                loss = torch.mean(self.model(data))
                print(f'{self.step}: {loss.item()}')
                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)

            acc_loss = acc_loss + (u_loss/self.gradient_accumulate_every)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                batches = self.batch_size
                og_img = next(self.dl).cuda()
                xt, direct_recons, all_images = self.ema_model.module.sample(batch_size=batches, img=og_img)

                og_img = (og_img + 1) * 0.5
                utils.save_image(og_img, str(self.results_folder / f'sample-og-{milestone}.png'), nrow=6)

                all_images = (all_images + 1) * 0.5
                utils.save_image(all_images, str(self.results_folder / f'sample-recon-{milestone}.png'), nrow = 6)

                direct_recons = (direct_recons + 1) * 0.5
                utils.save_image(direct_recons, str(self.results_folder / f'sample-direct_recons-{milestone}.png'), nrow=6)

                xt = (xt + 1) * 0.5
                utils.save_image(xt, str(self.results_folder / f'sample-xt-{milestone}.png'),
                                 nrow=6)

                acc_loss = acc_loss/(self.save_and_sample_every+1)

                print(f'Mean of last {self.step}: {acc_loss}')
                acc_loss=0

                self.save()

            self.step += 1

        print('training completed')


    def fid_distance_decrease_from_manifold(self, fid_func, start=0, end=1000):

        from pytorch_msssim import ssim

        all_samples = []
        dataset = self.ds

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = torch.unsqueeze(img, 0).cuda()
            if idx > start:
                all_samples.append(img[0])
            if idx % 1000 == 0:
                print(idx)
            if end != None:
                if idx == end:
                    print(idx)
                    break

        all_samples = torch.stack(all_samples)

        blurred_samples = None
        original_sample = None
        deblurred_samples = None
        direct_deblurred_samples = None

        sanity_check = 1

        cnt=0
        while(cnt < all_samples.shape[0]):
            og_x = all_samples[cnt: cnt + 200]
            og_x = og_x.cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            og_img = og_x
            print(og_img.shape)
            X_0s, X_ts = self.ema_model.module.all_sample(batch_size=og_img.shape[0], img=og_img, times=None)

            og_img = og_img.to('cpu')
            blurry_imgs = X_ts[0].to('cpu')
            deblurry_imgs = X_0s[-1].to('cpu')
            direct_deblurry_imgs = X_0s[0].to('cpu')

            og_img = og_img.repeat(1, 3 // og_img.shape[1], 1, 1)
            blurry_imgs = blurry_imgs.repeat(1, 3 // blurry_imgs.shape[1], 1, 1)
            deblurry_imgs = deblurry_imgs.repeat(1, 3 // deblurry_imgs.shape[1], 1, 1)
            direct_deblurry_imgs = direct_deblurry_imgs.repeat(1, 3 // direct_deblurry_imgs.shape[1], 1, 1)

            og_img = (og_img + 1) * 0.5
            blurry_imgs = (blurry_imgs + 1) * 0.5
            deblurry_imgs = (deblurry_imgs + 1) * 0.5
            direct_deblurry_imgs = (direct_deblurry_imgs + 1) * 0.5

            if cnt == 0:
                print(og_img.shape)
                print(blurry_imgs.shape)
                print(deblurry_imgs.shape)
                print(direct_deblurry_imgs.shape)

                if sanity_check:
                    folder = './sanity_check/'
                    create_folder(folder)

                    san_imgs = og_img[0: 32]
                    utils.save_image(san_imgs,str(folder + f'sample-og.png'), nrow=6)

                    san_imgs = blurry_imgs[0: 32]
                    utils.save_image(san_imgs, str(folder + f'sample-xt.png'), nrow=6)

                    san_imgs = deblurry_imgs[0: 32]
                    utils.save_image(san_imgs, str(folder + f'sample-recons.png'), nrow=6)

                    san_imgs = direct_deblurry_imgs[0: 32]
                    utils.save_image(san_imgs, str(folder + f'sample-direct-recons.png'), nrow=6)


            if blurred_samples is None:
                blurred_samples = blurry_imgs
            else:
                blurred_samples = torch.cat((blurred_samples, blurry_imgs), dim=0)


            if original_sample is None:
                original_sample = og_img
            else:
                original_sample = torch.cat((original_sample, og_img), dim=0)


            if deblurred_samples is None:
                deblurred_samples = deblurry_imgs
            else:
                deblurred_samples = torch.cat((deblurred_samples, deblurry_imgs), dim=0)


            if direct_deblurred_samples is None:
                direct_deblurred_samples = direct_deblurry_imgs
            else:
                direct_deblurred_samples = torch.cat((direct_deblurred_samples, direct_deblurry_imgs), dim=0)

            cnt += og_img.shape[0]

        print(blurred_samples.shape)
        print(original_sample.shape)
        print(deblurred_samples.shape)
        print(direct_deblurred_samples.shape)

        fid_blur = fid_func(samples=[original_sample, blurred_samples])
        rmse_blur = torch.sqrt(torch.mean( (original_sample - blurred_samples)**2 ))
        ssim_blur = ssim(original_sample, blurred_samples, data_range=1, size_average=True)
        print(f'The FID of blurry images with original image is {fid_blur}')
        print(f'The RMSE of blurry images with original image is {rmse_blur}')
        print(f'The SSIM of blurry images with original image is {ssim_blur}')

        fid_deblur = fid_func(samples=[original_sample, deblurred_samples])
        rmse_deblur = torch.sqrt(torch.mean((original_sample - deblurred_samples) ** 2))
        ssim_deblur = ssim(original_sample, deblurred_samples, data_range=1, size_average=True)
        print(f'The FID of deblurred images with original image is {fid_deblur}')
        print(f'The RMSE of deblurred images with original image is {rmse_deblur}')
        print(f'The SSIM of deblurred images with original image is {ssim_deblur}')

        print(f'Hence the improvement in FID using sampling is {fid_blur - fid_deblur}')

        fid_direct_deblur = fid_func(samples=[original_sample, direct_deblurred_samples])
        rmse_direct_deblur = torch.sqrt(torch.mean((original_sample - direct_deblurred_samples) ** 2))
        ssim_direct_deblur = ssim(original_sample, direct_deblurred_samples, data_range=1, size_average=True)
        print(f'The FID of direct deblurred images with original image is {fid_direct_deblur}')
        print(f'The RMSE of direct deblurred images with original image is {rmse_direct_deblur}')
        print(f'The SSIM of direct deblurred images with original image is {ssim_direct_deblur}')

        print(f'Hence the improvement in FID using direct sampling is {fid_blur - fid_direct_deblur}')

    def sample_as_a_mean_blur_torch_gmm_ablation(self, torch_gmm, siz=2, ch=3, clusters=10, sample_at=6, noise=0):

        flatten = nn.Flatten()
        all_samples = None
        batch_size = 100

        dl = data.DataLoader(self.ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16,
                             drop_last=True)

        for i, img in enumerate(dl, 0):
            print(img.shape)
            img = self.ema_model.module.opt(img.cuda(), t=sample_at)
            img = F.interpolate(img, size=siz, mode='area')
            img = flatten(img).cuda()
            if all_samples is None:
                all_samples = img
            else:
                all_samples = torch.cat((all_samples, img), dim=0)

        all_samples = all_samples.cuda()
        print(all_samples.shape)

        model = torch_gmm(num_components=clusters, trainer_params=dict(gpus=1), covariance_type='full',
                          convergence_tolerance=0.001, batch_size=batch_size)
        model.fit(all_samples)

        num_samples = 6400
        og_x = model.sample(num_datapoints=num_samples)
        og_x = og_x.cuda()
        og_x = og_x.reshape(num_samples, 3, siz, siz)
        og_x = F.interpolate(og_x, size=128, mode='nearest-exact')
        og_x = og_x.type(torch.cuda.FloatTensor)

        xt_folder = f'{self.results_folder}_xt'
        create_folder(xt_folder)

        out_folder = f'{self.results_folder}_out'
        create_folder(out_folder)

        direct_recons_folder = f'{self.results_folder}_dir_recons'
        create_folder(direct_recons_folder)

        cnt = 0
        bs = 64
        for j in range(100):
            og_img = og_x[j * bs: j * bs + bs]
            print(og_img.shape)
            og_img = og_img.expand(bs, ch, 128, 128)
            og_img = og_img.type(torch.cuda.FloatTensor)

            print(og_img.shape)
            xt, direct_recons, all_images = self.ema_model.module.gen_sample(batch_size=bs, img=og_img, noise_level=noise)
                                                                            

            for i in range(all_images.shape[0]):
                utils.save_image((all_images[i] + 1) * 0.5,
                                 str(f'{out_folder}/' + f'sample-x0-{cnt}.png'))

                utils.save_image((xt[i] + 1) * 0.5,
                                 str(f'{xt_folder}/' + f'sample-x0-{cnt}.png'))

                utils.save_image((direct_recons[i] + 1) * 0.5,
                                 str(f'{direct_recons_folder}/' + f'sample-x0-{cnt}.png'))

                cnt += 1
