import math
import copy
from torch import nn
import torch.nn.functional as F
import torch
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils

from einops import rearrange

from PIL import Image
from torch import linalg as LA

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


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

import numpy as np

def spiral_cw(A):
    A = np.array(A)
    out = []
    while(A.size):
        out.append(A[0])        # take first row
        A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
    return np.concatenate(out)


def spiral_ccw(A):
    A = np.array(A)
    out = []
    while(A.size):
        out.append(A[0][::-1])    # first row reversed
        A = A[1:][::-1].T         # cut off first row and rotate clockwise
    return np.concatenate(out)

def base_spiral(nrow, ncol):
    return spiral_ccw(np.arange(nrow*ncol).reshape(nrow,ncol))[::-1]

def to_spiral(A):
    A = np.array(A)
    B = np.empty_like(A)
    B.flat[base_spiral(*A.shape)] = A.flat
    return B

def from_spiral(A):
    A = np.array(A)
    return A.flat[base_spiral(*A.shape)].reshape(A.shape)

def to_spiral(A):
    A = np.array(A)
    B = np.empty_like(A)
    B.flat[base_spiral(*A.shape)] = A.flat
    return B

def from_spiral(A):
    A = np.array(A)
    return A.flat[base_spiral(*A.shape)].reshape(A.shape)




def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


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
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')

        h = self.net(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(
            self,
            dim,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            with_time_emb=True,
            residual=False
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
                ConvNextBlock(dim_in, dim_out, time_emb_dim=time_dim, norm=ind != 0),
                ConvNextBlock(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                ConvNextBlock(dim_in, dim_in, time_emb_dim=time_dim),
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


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            defade_fn,
            *,
            image_size,
            device_of_kernel,
            channels=3,
            timesteps=1000,
            loss_type='l1',
            start_fade_factor=0.1,
            fade_routine='Incremental',
            train_routine='Final',
            sampling_routine='default'
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.defade_fn = defade_fn
        self.device_of_kernel = device_of_kernel

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.start_fade_factor = start_fade_factor
        self.fade_routine = fade_routine

        self.fade_factors = self.get_fade_factors()
        self.train_routine = train_routine
        self.sampling_routine = sampling_routine

    def get_fade_factors(self):
        fade_factors = []
        for i in range(self.num_timesteps):
            if self.fade_routine == 'Incremental':
                fade_factors.append(1 - self.start_fade_factor * (i + 1))
            elif self.fade_routine == 'Constant':
                fade_factors.append(1 - self.start_fade_factor)
            elif self.fade_routine == 'Spiral':
                A = np.arange(32 * 32).reshape(32, 32)
                spiral = to_spiral(A)
                k = spiral > i
                k = torch.tensor(k).float()
                fade_factors.append(k.cuda())
            elif self.fade_routine == 'Spiral_2':
                A = np.arange(32 * 32).reshape(32, 32)
                spiral = to_spiral(A)
                k = spiral > i
                k = torch.tensor(k).float()
                fade_factors.append(k.cuda())


        return fade_factors

    @torch.no_grad()
    def sample(self, batch_size=16, img=None, t=None):

        if t is None:
            t = self.num_timesteps
        for i in range(t):
            with torch.no_grad():
                img = self.fade_factors[i] * img


        if self.fade_routine == 'Spiral_2':
            new_mean = torch.rand((img.shape[0], 3))
            new_mean = new_mean.unsqueeze(2).repeat(1, 1, img.shape[2])
            new_mean = new_mean.unsqueeze(3).repeat(1, 1, 1, img.shape[3]).cuda()

        for i in range(t):
            with torch.no_grad():
                if self.fade_routine == 'Spiral_2':
                    img = self.fade_factors[i] * img + new_mean * (torch.ones_like(self.fade_factors[i]) - self.fade_factors[i])
                else:
                    img = self.fade_factors[i] * img

        xt = img
        direct_recons = None
        while t:
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x = self.defade_fn(img, step)

            if "Final" in self.train_routine:
                if direct_recons is None:
                    direct_recons = x

                if self.sampling_routine == 'default':
                    for i in range(t - 1):
                        with torch.no_grad():
                            x = self.fade_factors[i] * x

                elif self.sampling_routine == 'x0_step_down':
                    x_times = x
                    x_times_sub_1 = x

                    for i in range(t):
                        with torch.no_grad():
                            x_times_sub_1 = x_times
                            x_times = self.fade_factors[i] * x_times

                    x = img - x_times + x_times_sub_1

                elif self.sampling_routine == 'x0_step_down_spiral_2_fix':
                    x_times = x
                    x_times_sub_1 = x

                    for i in range(t):
                        with torch.no_grad():
                            x_times_sub_1 = x_times
                            x_times = self.fade_factors[i] * x_times + new_mean * (
                                        torch.ones_like(self.fade_factors[i]) - self.fade_factors[i])

                    x = img - x_times + x_times_sub_1

                elif self.sampling_routine == 'x0_step_down_spiral_2_rand':
                    x_times = x
                    x_times_sub_1 = x

                    for i in range(t):
                        new_mean = torch.rand((img.shape[0], 3))
                        new_mean = new_mean.unsqueeze(2).repeat(1, 1, img.shape[2])
                        new_mean = new_mean.unsqueeze(3).repeat(1, 1, 1, img.shape[3]).cuda()

                        with torch.no_grad():
                            x_times_sub_1 = x_times
                            x_times = self.fade_factors[i] * x_times + new_mean * (
                                        torch.ones_like(self.fade_factors[i]) - self.fade_factors[i])

                    x = img - x_times + x_times_sub_1

            elif self.train_routine == 'Step':
                if direct_recons is None:
                    direct_recons = x

            elif self.train_routine == 'Gradient_norm':
                if direct_recons is None:
                    direct_recons = img - x
                x = img - x

            img = x
            t = t - 1

        return xt, direct_recons, img

    @torch.no_grad()
    def all_sample(self, batch_size=16, img=None, t=None, times=None):

        if t is None:
            t = self.num_timesteps
        if times is None:
            times = t

        if self.fade_routine == 'Spiral_2':
            new_mean = torch.rand((img.shape[0], 3))
            new_mean = new_mean.unsqueeze(2).repeat(1, 1, img.shape[2])
            new_mean = new_mean.unsqueeze(3).repeat(1, 1, 1, img.shape[3]).cuda()

        for i in range(t):
            with torch.no_grad():
                if self.fade_routine == 'Spiral_2':
                    img = self.fade_factors[i] * img + new_mean * (
                                torch.ones_like(self.fade_factors[i]) - self.fade_factors[i])
                else:
                    img = self.fade_factors[i] * img


        x0_list = []
        xt_list = []

        while times:
            step = torch.full((batch_size,), times - 1, dtype=torch.long).cuda()
            x = self.defade_fn(img, step)
            x0_list.append(x)

            if "Final" in self.train_routine:
                if self.sampling_routine == 'default':
                    print("Normal")

                    x_times_sub_1 = x
                    for i in range(times - 1):
                        with torch.no_grad():
                            x_times_sub_1 = self.fade_factors[i] * x_times_sub_1

                    x = x_times_sub_1

                elif self.sampling_routine == 'x0_step_down':
                    print("x0_step_down")

                    x_times = x
                    for i in range(times):
                        with torch.no_grad():
                            x_times = self.fade_factors[i] * x_times

                    x_times_sub_1 = x
                    for i in range(times - 1):
                        with torch.no_grad():
                            x_times_sub_1 = self.fade_factors[i] * x_times_sub_1

                    x = img - x_times + x_times_sub_1

                elif self.sampling_routine == 'x0_step_down_spiral_2_fix':
                    x_times = x
                    x_times_sub_1 = x

                    for i in range(times):
                        with torch.no_grad():
                            x_times_sub_1 = x_times
                            x_times = self.fade_factors[i] * x_times + new_mean * (
                                        torch.ones_like(self.fade_factors[i]) - self.fade_factors[i])

                    x = img - x_times + x_times_sub_1

                elif self.sampling_routine == 'x0_step_down_spiral_2_rand':
                    x_times = x
                    x_times_sub_1 = x

                    for i in range(times):
                        new_mean = torch.rand((img.shape[0], 3))
                        new_mean = new_mean.unsqueeze(2).repeat(1, 1, img.shape[2])
                        new_mean = new_mean.unsqueeze(3).repeat(1, 1, 1, img.shape[3]).cuda()

                        with torch.no_grad():
                            x_times_sub_1 = x_times
                            x_times = self.fade_factors[i] * x_times + new_mean * (
                                        torch.ones_like(self.fade_factors[i]) - self.fade_factors[i])

                    x = img + x_times_sub_1 - img #- x_times

                elif self.sampling_routine == 'no_time_embed':
                    x = x
                    for i in range(100):
                        with torch.no_grad():
                            x = self.fade_factors[i] * x

            elif self.train_routine == 'Gradient_norm':
                x = img - 0.1 * x
                for i in range(10):
                    with torch.no_grad():
                        x = self.fade_factors[i] * x

            img = x
            xt_list.append(img)
            times = times - 1

        return x0_list, xt_list

    def q_sample(self, x_start, t):

        if self.fade_routine == 'Spiral':
            choose_fade = []
            for img_index in range(t.shape[0]):
                choose_fade.append(x_start[img_index,:] * self.fade_factors[t[img_index]] )

            choose_fade = torch.stack(choose_fade)
            return choose_fade

        elif self.fade_routine == 'Spiral_2':

            choose_fade = []
            for img_index in range(t.shape[0]):
                new_mean = torch.rand((1, 3))
                new_mean = new_mean.unsqueeze(2).repeat(1, 1, x_start.shape[2])
                new_mean = new_mean.unsqueeze(3).repeat(1, 1, 1, x_start.shape[3]).cuda()

                cf = x_start[img_index,:] * self.fade_factors[t[img_index]] + new_mean * (torch.ones_like(self.fade_factors[t[img_index]]) - self.fade_factors[t[img_index]])
                choose_fade.append(cf[0,:])

            choose_fade = torch.stack(choose_fade)
            return choose_fade

        else:
            max_iters = torch.max(t)
            all_fades = []
            x = x_start
            for i in range(max_iters + 1):
                with torch.no_grad():
                    x = self.fade_factors[i] * x
                    all_fades.append(x)

            all_fades = torch.stack(all_fades)

            choose_fade = []
            for step in range(t.shape[0]):
                if step != -1:
                    choose_fade.append(all_fades[t[step], step])
                else:
                    choose_fade.append(x_start[step])

            choose_fade = torch.stack(choose_fade)
            return choose_fade


    def p_losses(self, x_start, t):
        if self.train_routine == 'Final':
            x_fade = self.q_sample(x_start=x_start, t=t)
            x_recon = self.defade_fn(x_fade, t)

            if self.loss_type == 'l1':
                loss = (x_start - x_recon).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()

        elif self.train_routine == 'Gradient_norm':
            x_fade = self.q_sample(x_start=x_start, t=t)
            grad_pred = self.defade_fn(x_fade, t)
            gradient = (x_fade - x_start)
            norm = LA.norm(gradient, dim=(1, 2, 3), keepdim=True)
            gradient_norm = gradient / (norm + 1e-5)

            if self.loss_type == 'l1':
                loss = (gradient_norm - grad_pred).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(gradient_norm, grad_pred)
            else:
                raise NotImplementedError()

        elif self.train_routine == 'Step':
            x_fade = self.q_sample(x_start=x_start, t=t)
            x_fade_sub = self.q_sample(x_start=x_start, t=t - 1)
            x_blur_sub_pred = self.defade_fn(x_fade, t)

            if self.loss_type == 'l1':
                loss = (x_fade_sub - x_blur_sub_pred).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_fade_sub, x_blur_sub_pred)
            else:
                raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
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


class Dataset_Cifar10(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.Resize(image_size),
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


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            folder,
            *,
            ema_decay=0.995,
            image_size=128,
            train_batch_size=32,
            train_lr=2e-5,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            fp16=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=1000,
            results_folder='./results',
            load_path=None,
            dataset=None
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        if dataset == 'cifar10':
            self.ds = Dataset_Cifar10(folder, image_size)
        else:
            self.ds = Dataset(folder, image_size)
        self.dl = cycle(data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt,
                                                                    opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.reset_parameters()

        if load_path is not None:
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

    def add_title(self, path, title):
        import cv2
        import numpy as np

        img1 = cv2.imread(path)
        black = [0, 0, 0]
        constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        height = 20
        violet = np.zeros((height, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)

        vcat = cv2.vconcat((violet, constant))

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(vcat, str(title), (violet.shape[1] // 2, height - 2), font, 0.5, (0, 0, 0), 1, 0)
        cv2.imwrite(path, vcat)

    def train(self):
        backwards = partial(loss_backwards, self.fp16)

        acc_loss = 0
        while self.step < self.train_num_steps:
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()
                loss = self.model(data)
                print(f'{self.step}: {loss.item()}')
                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)

            acc_loss = acc_loss + (u_loss / self.gradient_accumulate_every)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                batches = self.batch_size
                og_img = next(self.dl).cuda()
                xt, direct_recons, all_images = self.ema_model.sample(batch_size=batches, faded_recon_sample=og_img)

                og_img = (og_img + 1) * 0.5
                utils.save_image(og_img, str(self.results_folder / f'sample-og-{milestone}.png'), nrow=6)

                all_images = (all_images + 1) * 0.5
                utils.save_image(all_images, str(self.results_folder / f'sample-recon-{milestone}.png'), nrow=6)

                direct_recons = (direct_recons + 1) * 0.5
                utils.save_image(direct_recons, str(self.results_folder / f'sample-direct_recons-{milestone}.png'),
                                 nrow=6)

                xt = (xt + 1) * 0.5
                utils.save_image(xt, str(self.results_folder / f'sample-xt-{milestone}.png'),
                                 nrow=6)

                acc_loss = acc_loss / (self.save_and_sample_every + 1)
                print(f'Mean of last {self.step}: {acc_loss}')
                acc_loss = 0

                self.save()

            self.step += 1

        print('training completed')

    def test_from_data(self, extra_path, s_times=None):
        batches = self.batch_size
        og_img = next(self.dl).cuda()
        x0_list, xt_list = self.ema_model.all_sample(batch_size=batches, faded_recon_sample=og_img, times=s_times)

        og_img = (og_img + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'og-{extra_path}.png'), nrow=6)

        import imageio
        frames_t = []
        frames_0 = []

        for i in range(len(x0_list)):
            print(i)

            x_0 = x0_list[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), str(i))
            frames_0.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png')))

            x_t = xt_list[i]
            all_images = (x_t + 1) * 0.5
            utils.save_image(all_images, str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), str(i))
            frames_t.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png')))

        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), frames_0)
        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), frames_t)

    def test_with_mixup(self, extra_path):
        batches = self.batch_size
        og_img_1 = next(self.dl).cuda()
        og_img_2 = next(self.dl).cuda()
        og_img = (og_img_1 + og_img_2) / 2

        x0_list, xt_list = self.ema_model.all_sample(batch_size=batches, faded_recon_sample=og_img)

        og_img_1 = (og_img_1 + 1) * 0.5
        utils.save_image(og_img_1, str(self.results_folder / f'og1-{extra_path}.png'), nrow=6)

        og_img_2 = (og_img_2 + 1) * 0.5
        utils.save_image(og_img_2, str(self.results_folder / f'og2-{extra_path}.png'), nrow=6)

        og_img = (og_img + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'og-{extra_path}.png'), nrow=6)

        frames_t = []
        frames_0 = []

        for i in range(len(x0_list)):
            print(i)
            x_0 = x0_list[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), str(i))
            frames_0.append(Image.open(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png')))

            x_t = xt_list[i]
            all_images = (x_t + 1) * 0.5
            utils.save_image(all_images, str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), str(i))
            frames_t.append(Image.open(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png')))

        frame_one = frames_0[0]
        frame_one.save(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), format="GIF", append_images=frames_0,
                       save_all=True, duration=100, loop=0)

        frame_one = frames_t[0]
        frame_one.save(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), format="GIF", append_images=frames_t,
                       save_all=True, duration=100, loop=0)

    def test_from_random(self, extra_path):
        batches = self.batch_size
        og_img = next(self.dl).cuda()
        og_img = og_img * 0.9  # torch.randn_like(og_img) + 0.1
        x0_list, xt_list = self.ema_model.all_sample(batch_size=batches, faded_recon_sample=og_img)

        og_img = (og_img + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'og-{extra_path}.png'), nrow=6)

        frames_t_names = []
        frames_0_names = []

        for i in range(len(x0_list)):
            print(i)

            x_0 = x0_list[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), str(i))
            frames_0_names.append(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'))

            x_t = xt_list[i]
            all_images = (x_t + 1) * 0.5
            utils.save_image(all_images, str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), str(i))
            frames_t_names.append(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'))

        import imageio
        frames_0 = []
        frames_t = []
        for i in range(len(x0_list)):
            print(i)
            frames_0.append(imageio.imread(frames_0_names[i]))
            frames_t.append(imageio.imread(frames_t_names[i]))

        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), frames_0)
        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), frames_t)
