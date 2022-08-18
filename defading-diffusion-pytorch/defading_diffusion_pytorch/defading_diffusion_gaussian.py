import math
import copy
import torch
from torch import nn
import torch.nn.functional as func
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils

from einops import rearrange

import torchgeometry as tgm
import os
import errno
from PIL import Image
from pytorch_msssim import ssim
import cv2
import numpy as np
import imageio

# from torch.utils.tensorboard import SummaryWriter

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


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def cycle(dl):
    while True:
        for inputs in dl:
            yield inputs


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


class EMA:
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
            kernel_std=0.1,
            initial_mask=11,
            fade_routine='Incremental',
            sampling_routine='default',
            discrete=False
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.defade_fn = defade_fn
        self.device_of_kernel = device_of_kernel
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.kernel_std = kernel_std
        self.initial_mask = initial_mask
        self.fade_routine = fade_routine
        self.fade_kernels = self.get_kernels()
        self.sampling_routine = sampling_routine
        self.discrete = discrete

    def get_fade_kernel(self, dims, std):
        fade_kernel = tgm.image.get_gaussian_kernel2d(dims, std)
        fade_kernel = fade_kernel / torch.max(fade_kernel)
        fade_kernel = torch.ones_like(fade_kernel) - fade_kernel
        # if self.device_of_kernel == 'cuda':
        #     fade_kernel = fade_kernel.cuda()
        fade_kernel = fade_kernel[1:, 1:]
        return fade_kernel

    def get_kernels(self):
        kernels = []
        for i in range(self.num_timesteps):
            if self.fade_routine == 'Incremental':
                kernels.append(self.get_fade_kernel((self.image_size + 1, self.image_size + 1),
                                                    (self.kernel_std * (i + self.initial_mask),
                                                     self.kernel_std * (i + self.initial_mask))))
            elif self.fade_routine == 'Constant':
                kernels.append(self.get_fade_kernel(
                    (self.image_size + 1, self.image_size + 1),
                    (self.kernel_std, self.kernel_std)))
            elif self.fade_routine == 'Random_Incremental':
                kernels.append(self.get_fade_kernel((2 * self.image_size + 1, 2 * self.image_size + 1),
                                                    (self.kernel_std * (i + self.initial_mask),
                                                     self.kernel_std * (i + self.initial_mask))))
        return torch.stack(kernels)

    @torch.no_grad()
    def sample(self, batch_size=16, faded_recon_sample=None, t=None):

        rand_fade_kernels = None
        sample_device = faded_recon_sample.device
        if 'Random' in self.fade_routine:
            rand_fade_kernels = []
            rand_x = torch.randint(0, self.image_size + 1, (batch_size,), device=sample_device).long()
            rand_y = torch.randint(0, self.image_size + 1, (batch_size,), device=sample_device).long()
            for i in range(batch_size):
                rand_fade_kernels.append(torch.stack(
                    [self.fade_kernels[j][rand_x[i]:rand_x[i] + self.image_size,
                     rand_y[i]:rand_y[i] + self.image_size] for j in range(len(self.fade_kernels))]))
            rand_fade_kernels = torch.stack(rand_fade_kernels)
        if t is None:
            t = self.num_timesteps

        for i in range(t):
            with torch.no_grad():
                if rand_fade_kernels is not None:
                    faded_recon_sample = torch.stack([rand_fade_kernels[:, i].to(sample_device),
                                                      rand_fade_kernels[:, i].to(sample_device),
                                                      rand_fade_kernels[:, i].to(sample_device)],
                                                     1) * faded_recon_sample
                else:
                    faded_recon_sample = self.fade_kernels[i].to(sample_device) * faded_recon_sample

        if self.discrete:
            faded_recon_sample = (faded_recon_sample + 1) * 0.5
            faded_recon_sample = (faded_recon_sample * 255)
            faded_recon_sample = faded_recon_sample.int().float() / 255
            faded_recon_sample = faded_recon_sample * 2 - 1

        xt = faded_recon_sample
        direct_recons = None
        recon_sample = None

        while t:
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            recon_sample = self.defade_fn(faded_recon_sample, step)

            if direct_recons is None:
                direct_recons = recon_sample

            if self.sampling_routine == 'default':
                for i in range(t - 1):
                    with torch.no_grad():
                        if rand_fade_kernels is not None:
                            recon_sample = torch.stack([rand_fade_kernels[:, i].to(sample_device),
                                                        rand_fade_kernels[:, i].to(sample_device),
                                                        rand_fade_kernels[:, i].to(sample_device)], 1) * recon_sample
                        else:
                            recon_sample = self.fade_kernels[i].to(sample_device) * recon_sample
                faded_recon_sample = recon_sample

            elif self.sampling_routine == 'x0_step_down':
                for i in range(t):
                    with torch.no_grad():
                        recon_sample_sub_1 = recon_sample
                        if rand_fade_kernels is not None:
                            recon_sample = torch.stack([rand_fade_kernels[:, i].to(sample_device),
                                                        rand_fade_kernels[:, i].to(sample_device),
                                                        rand_fade_kernels[:, i].to(sample_device)], 1) * recon_sample
                        else:
                            recon_sample = self.fade_kernels[i].to(sample_device) * recon_sample

                faded_recon_sample = faded_recon_sample - recon_sample + recon_sample_sub_1

            recon_sample = faded_recon_sample
            t -= 1

        return xt, direct_recons, recon_sample

    @torch.no_grad()
    def all_sample(self, batch_size=16, faded_recon_sample=None, t=None, times=None):
        rand_fade_kernels = None
        sample_device = faded_recon_sample.device
        if 'Random' in self.fade_routine:
            rand_fade_kernels = []
            rand_x = torch.randint(0, self.image_size + 1, (batch_size,), device=faded_recon_sample.device).long()
            rand_y = torch.randint(0, self.image_size + 1, (batch_size,), device=faded_recon_sample.device).long()
            for i in range(batch_size, ):
                rand_fade_kernels.append(torch.stack(
                    [self.fade_kernels[j][rand_x[i]:rand_x[i] + self.image_size,
                     rand_y[i]:rand_y[i] + self.image_size] for j in range(len(self.fade_kernels))]))
            rand_fade_kernels = torch.stack(rand_fade_kernels)
        if t is None:
            t = self.num_timesteps
        if times is None:
            times = t

        for i in range(t):
            with torch.no_grad():
                if 'Random' in self.fade_routine:
                    faded_recon_sample = torch.stack([rand_fade_kernels[:, i].to(sample_device),
                                                      rand_fade_kernels[:, i].to(sample_device),
                                                      rand_fade_kernels[:, i].to(sample_device)], 1) * faded_recon_sample
                else:
                    faded_recon_sample = self.fade_kernels[i].to(sample_device) * faded_recon_sample

        if self.discrete:
            faded_recon_sample = (faded_recon_sample + 1) * 0.5
            faded_recon_sample = (faded_recon_sample * 255)
            faded_recon_sample = faded_recon_sample.int().float() / 255
            faded_recon_sample = faded_recon_sample * 2 - 1

        x0_list = []
        xt_list = []

        while times:
            step = torch.full((batch_size,), times - 1, dtype=torch.long).cuda()
            recon_sample = self.defade_fn(faded_recon_sample, step)
            x0_list.append(recon_sample)

            if self.sampling_routine == 'default':
                for i in range(times - 1):
                    with torch.no_grad():
                        if rand_fade_kernels is not None:
                            recon_sample = torch.stack([rand_fade_kernels[:, i].to(sample_device),
                                                        rand_fade_kernels[:, i].to(sample_device),
                                                        rand_fade_kernels[:, i].to(sample_device)], 1) * recon_sample
                        else:
                            recon_sample = self.fade_kernels[i].to(sample_device) * recon_sample
                faded_recon_sample = recon_sample

            elif self.sampling_routine == 'x0_step_down':
                for i in range(times):
                    with torch.no_grad():
                        recon_sample_sub_1 = recon_sample
                        if rand_fade_kernels is not None:
                            recon_sample = torch.stack([rand_fade_kernels[:, i].to(sample_device),
                                                        rand_fade_kernels[:, i].to(sample_device),
                                                        rand_fade_kernels[:, i].to(sample_device)], 1) * recon_sample
                        else:
                            recon_sample = self.fade_kernels[i].to(sample_device) * recon_sample
                faded_recon_sample = faded_recon_sample - recon_sample + recon_sample_sub_1

            xt_list.append(faded_recon_sample)
            times -= 1

        return x0_list, xt_list

    def q_sample(self, x_start, t):
        with torch.no_grad():
            rand_fade_kernels = None
            if 'Random' in self.fade_routine:
                rand_fade_kernels = []
                rand_x = torch.randint(0, self.image_size + 1, (x_start.size(0),), device=x_start.device).long()
                rand_y = torch.randint(0, self.image_size + 1, (x_start.size(0),), device=x_start.device).long()
                for i in range(x_start.size(0),):
                    rand_fade_kernels.append(torch.stack(
                        [self.fade_kernels[j][rand_x[i]:rand_x[i] + self.image_size,
                         rand_y[i]:rand_y[i] + self.image_size] for j in range(len(self.fade_kernels))]))
                rand_fade_kernels = torch.stack(rand_fade_kernels)
        max_iters = torch.max(t)
        all_fades = []
        x = x_start
        for i in range(max_iters + 1):
            with torch.no_grad():
                if rand_fade_kernels is not None:
                    x = torch.stack([rand_fade_kernels[:, i],
                                     rand_fade_kernels[:, i],
                                     rand_fade_kernels[:, i]], 1) * x
                else:
                    x = self.fade_kernels[i] * x
                all_fades.append(x)

        all_fades = torch.stack(all_fades)

        choose_fade = []
        for step in range(t.shape[0]):
            if step != -1:
                choose_fade.append(all_fades[t[step], step])
            else:
                choose_fade.append(x_start[step])
        choose_fade = torch.stack(choose_fade)
        if self.discrete:
            choose_fade = (choose_fade + 1) * 0.5
            choose_fade = (choose_fade * 255)
            choose_fade = choose_fade.int().float() / 255
            choose_fade = choose_fade * 2 - 1
        return choose_fade

    def p_losses(self, x_start, t):
        x_fade = self.q_sample(x_start=x_start, t=t)
        x_recon = self.defade_fn(x_fade, t)

        if self.loss_type == 'l1':
            loss = (x_start - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = func.mse_loss(x_start, x_recon)
        else:
            raise NotImplementedError()
        return loss

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        self.fade_kernels = self.fade_kernels.to(device)
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


class DatasetCifar10(data.Dataset):
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


class DatasetCelebA(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
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


class DatasetCelebATest(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
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
            train_num_steps=700000,
            gradient_accumulate_every=2,
            fp16=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=10000,
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
        self.image_size = diffusion_model.module.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        if dataset == 'cifar10':
            self.ds = DatasetCifar10(folder, image_size)
        elif dataset == 'celebA':
            self.ds = DatasetCelebA(folder, image_size)
        elif dataset == 'celebA_test':
            self.ds = DatasetCelebATest(folder, image_size)
        else:
            self.ds = Dataset(folder, image_size)
        self.dl = cycle(
            data.DataLoader(self.ds,
                            batch_size=train_batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=16,
                            drop_last=True))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed for mixed precision training on'

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
        model_data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(model_data, str(self.results_folder / f'model.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        model_data = torch.load(load_path)

        self.step = model_data['step']
        self.model.load_state_dict(model_data['model'])
        self.ema_model.load_state_dict(model_data['ema'])

    @staticmethod
    def add_title(path, title):
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
        # writer = SummaryWriter()

        acc_loss = 0
        while self.step < self.train_num_steps:
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                inputs = next(self.dl).cuda()
                # loss = self.model(inputs)
                loss = torch.mean(self.model(inputs))
                print(f'{self.step}: {loss.item()}')
                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)
            # writer.add_scalar("Loss/train", loss.item(), self.step)
            acc_loss = acc_loss + (u_loss / self.gradient_accumulate_every)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                batches = self.batch_size
                og_img = next(self.dl).cuda()
                # xt, direct_recons, all_images = self.ema_model.sample(batch_size=batches, faded_recon_sample=og_img)
                xt, direct_recons, all_images = self.ema_model.module.sample(batch_size=batches, faded_recon_sample=og_img)
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
        x0_list, xt_list = self.ema_model.module.all_sample(batch_size=batches, faded_recon_sample=og_img, times=s_times)

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

        x0_list, xt_list = self.ema_model.module.all_sample(batch_size=batches, faded_recon_sample=og_img)

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
        og_img = og_img * 0.9
        x0_list, xt_list = self.ema_model.module.all_sample(batch_size=batches, faded_recon_sample=og_img)

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

        frames_0 = []
        frames_t = []
        for i in range(len(x0_list)):
            print(i)
            frames_0.append(imageio.imread(frames_0_names[i]))
            frames_t.append(imageio.imread(frames_t_names[i]))

        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), frames_0)
        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), frames_t)

    def controlled_direct_reconstruct(self, extra_path):
        batches = self.batch_size
        torch.manual_seed(0)
        og_img = next(self.dl).cuda()
        xt, direct_recons, all_images = self.ema_model.module.sample(batch_size=batches, faded_recon_sample=og_img)

        og_img = (og_img + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'sample-og-{extra_path}.png'), nrow=6)

        all_images = (all_images + 1) * 0.5
        utils.save_image(all_images, str(self.results_folder / f'sample-recon-{extra_path}.png'), nrow=6)

        direct_recons = (direct_recons + 1) * 0.5
        utils.save_image(direct_recons, str(self.results_folder / f'sample-direct_recons-{extra_path}.png'), nrow=6)

        xt = (xt + 1) * 0.5
        utils.save_image(xt, str(self.results_folder / f'sample-xt-{extra_path}.png'),
                         nrow=6)

        self.save()

    def fid_distance_decrease_from_manifold(self, fid_func, start=0, end=1000):

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
            if end is not None:
                if idx == end:
                    print(idx)
                    break

        all_samples = torch.stack(all_samples)
        blurred_samples = None
        original_sample = None
        deblurred_samples = None
        direct_deblurred_samples = None

        sanity_check = blurred_samples

        cnt = 0
        while cnt < all_samples.shape[0]:
            og_x = all_samples[cnt: cnt + 50]
            og_x = og_x.cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            og_img = og_x
            print(og_img.shape)
            x0_list, xt_list = self.ema_model.module.all_sample(batch_size=og_img.shape[0],
                                                         faded_recon_sample=og_img,
                                                         times=None)

            og_img = og_img.to('cpu')
            blurry_imgs = xt_list[0].to('cpu')
            deblurry_imgs = x0_list[-1].to('cpu')
            direct_deblurry_imgs = x0_list[0].to('cpu')

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
                    utils.save_image(san_imgs, str(folder + f'sample-og.png'), nrow=6)

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
        rmse_blur = torch.sqrt(torch.mean((original_sample - blurred_samples) ** 2))
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

    def paper_invert_section_images(self, s_times=None):

        cnt = 0
        for i in range(50):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            x0_list, xt_list = self.ema_model.module.all_sample(batch_size=batches,
                                                         faded_recon_sample=og_img,
                                                         times=s_times)
            og_img = (og_img + 1) * 0.5

            for j in range(og_img.shape[0]//3):
                original = og_img[j: j + 1]
                utils.save_image(original, str(self.results_folder / f'original_{cnt}.png'), nrow=3)

                direct_recons = x0_list[0][j: j + 1]
                direct_recons = (direct_recons + 1) * 0.5
                utils.save_image(direct_recons, str(self.results_folder / f'direct_recons_{cnt}.png'), nrow=3)

                sampling_recons = x0_list[-1][j: j + 1]
                sampling_recons = (sampling_recons + 1) * 0.5
                utils.save_image(sampling_recons, str(self.results_folder / f'sampling_recons_{cnt}.png'), nrow=3)

                blurry_image = xt_list[0][j: j + 1]
                blurry_image = (blurry_image + 1) * 0.5
                utils.save_image(blurry_image, str(self.results_folder / f'blurry_image_{cnt}.png'), nrow=3)

                blurry_image = cv2.imread(f'{self.results_folder}/blurry_image_{cnt}.png')
                direct_recons = cv2.imread(f'{self.results_folder}/direct_recons_{cnt}.png')
                sampling_recons = cv2.imread(f'{self.results_folder}/sampling_recons_{cnt}.png')
                original = cv2.imread(f'{self.results_folder}/original_{cnt}.png')

                black = [0, 0, 0]
                blurry_image = cv2.copyMakeBorder(blurry_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                direct_recons = cv2.copyMakeBorder(direct_recons, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                sampling_recons = cv2.copyMakeBorder(sampling_recons, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                original = cv2.copyMakeBorder(original, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)

                im_h = cv2.hconcat([blurry_image, direct_recons, sampling_recons, original])
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)

                cnt += 1

    def paper_showing_diffusion_images(self, s_times=None):

        cnt = 0
        to_show = [0, 1, 2, 4, 8, 16, 32, 64, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

        for i in range(100):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            x0_list, xt_list = self.ema_model.module.all_sample(batch_size=batches, faded_recon_sample=og_img, times=s_times)

            for k in range(xt_list[0].shape[0]):
                lst = []

                for j in range(len(xt_list)):
                    x_t = xt_list[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(x_t, str(self.results_folder / f'x_{len(xt_list)-j}_{cnt}.png'), nrow=1)
                    x_t = cv2.imread(f'{self.results_folder}/x_{len(xt_list)-j}_{cnt}.png')
                    if j in to_show:
                        lst.append(x_t)

                x_0 = x0_list[-1][k]
                x_0 = (x_0 + 1) * 0.5
                utils.save_image(x_0, str(self.results_folder / f'x_best_{cnt}.png'), nrow=1)
                x_0 = cv2.imread(f'{self.results_folder}/x_best_{cnt}.png')
                lst.append(x_0)
                im_h = cv2.hconcat(lst)
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)
                cnt += 1

    def test_from_data_save_results(self):
        batch_size = 100
        dl = data.DataLoader(self.ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16,
                             drop_last=True)

        all_samples = None

        for i, img in enumerate(dl, 0):
            print(i)
            print(img.shape)
            if all_samples is None:
                all_samples = img
            else:
                all_samples = torch.cat((all_samples, img), dim=0)

            # break

        # create_folder(f'{self.results_folder}/')
        blurred_samples = None
        original_sample = None
        deblurred_samples = None
        direct_deblurred_samples = None

        sanity_check = 1

        orig_folder = f'{self.results_folder}_orig/'
        create_folder(orig_folder)

        blur_folder = f'{self.results_folder}_blur/'
        create_folder(blur_folder)

        d_deblur_folder = f'{self.results_folder}_d_deblur/'
        create_folder(d_deblur_folder)

        deblur_folder = f'{self.results_folder}_deblur/'
        create_folder(deblur_folder)

        cnt = 0
        while cnt < all_samples.shape[0]:
            print(cnt)
            og_x = all_samples[cnt: cnt + 32]
            og_x = og_x.cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            og_img = og_x
            x0_list, xt_list = self.ema_model.module.all_sample(batch_size=og_img.shape[0], faded_recon_sample=og_img, times=None)

            og_img = og_img.to('cpu')
            blurry_imgs = xt_list[0].to('cpu')
            deblurry_imgs = x0_list[-1].to('cpu')
            direct_deblurry_imgs = x0_list[0].to('cpu')

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

        for i in range(blurred_samples.shape[0]):
            utils.save_image(original_sample[i], f'{orig_folder}{i}.png', nrow=1)
            utils.save_image(blurred_samples[i], f'{blur_folder}{i}.png', nrow=1)
            utils.save_image(deblurred_samples[i], f'{deblur_folder}{i}.png', nrow=1)
            utils.save_image(direct_deblurred_samples[i], f'{d_deblur_folder}{i}.png', nrow=1)