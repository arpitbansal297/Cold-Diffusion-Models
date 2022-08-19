import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import time

import torch.linalg

from torch.utils import data
from torchvision import transforms, utils
from torchvision import datasets

import numpy as np
from tqdm import tqdm
from einops import rearrange

import torchgeometry as tgm
import glob
import os
from torch import linalg as LA
from .utils import rgb2lab, lab2rgb

from scipy.ndimage import zoom as scizoom
from PIL import Image as PILImage
from kornia.color.gray import rgb_to_grayscale
import cv2


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


class ForwardProcessBase:
    
    def forward(self, x, i):
        pass

    @torch.no_grad()
    def reset_parameters(self, batch_size=32):
        pass


class GaussianBlur(ForwardProcessBase):

    def __init__(self, 
                 blur_routine='Constant', 
                 kernel_std=0.1,
                 kernel_size=3,
                 start_kernel_std=0.01, 
                 target_kernel_std=1.0,
                 num_timesteps=50,
                 channels=3,):
        assert blur_routine != 'Individual_Incremental'
        self.blur_routine = blur_routine
        self.kernel_std = kernel_std
        self.kernel_size = kernel_size
        self.start_kernel_std = start_kernel_std
        self.target_kernel_std = target_kernel_std
        self.num_timesteps = num_timesteps
        self.channels = channels
        self.device_of_kernel = 'cuda'
        self.kernels = self.get_kernels()


    def blur(self, dims, std):
        return tgm.image.get_gaussian_kernel2d(dims, std)

    def get_conv(self, dims, std):
        kernel = self.blur(dims, std)
        conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=dims, padding=int((dims[0]-1)/2), padding_mode='circular',
                         bias=False, groups=self.channels)
        with torch.no_grad():
            kernel = torch.unsqueeze(kernel, 0)
            kernel = torch.unsqueeze(kernel, 0)
            kernel = kernel.repeat(self.channels, 1, 1, 1)
            conv.weight = nn.Parameter(kernel)

        if self.device_of_kernel == 'cuda':
            conv = conv.cuda()

        return conv

    def get_kernels(self):
        kernels = []
        
        if self.blur_routine == 'Linear_Accum_Std':
            accum_std_list = torch.linspace(self.start_kernel_std, self.target_kernel_std, self.num_timesteps).tolist()
            self.kernel_std_list = [accum_std_list[0]]
            for i in range(1, len(accum_std_list)):
                self.kernel_std_list.append(np.sqrt(accum_std_list[i] ** 2 - accum_std_list[i-1] ** 2))
        
        if self.blur_routine == 'Linear_Dec_Std':
            std_scale_list = torch.linspace(1.0, self.start_kernel_std, self.num_timesteps)
            std_ratio = (self.target_kernel_std ** 2 / std_scale_list.square().sum()).sqrt()
            self.kernel_std_list = (std_scale_list * std_ratio).tolist()


        if self.blur_routine in ['Linear_Accum_Std', 'Linear_Dec_Std']:
            # size determine by two sigma
            self.kernel_size_list = []
            for i in range(len(self.kernel_std_list)):
                size = 2 * int(2 * self.kernel_std_list[i]) + 3
                self.kernel_size_list.append(size)

        for i in range(self.num_timesteps):
            if self.blur_routine == 'Incremental':
                kernels.append(self.get_conv((self.kernel_size, self.kernel_size), (self.kernel_std*(i+1), self.kernel_std*(i+1)) ) )
            elif self.blur_routine == 'Constant':
                kernels.append(self.get_conv((self.kernel_size, self.kernel_size), (self.kernel_std, self.kernel_std) ) )
            elif self.blur_routine in ['Linear_Accum_Std', 'Linear_Dec_Std']:
                kernels.append(self.get_conv((self.kernel_size_list[i], self.kernel_size_list[i]), (self.kernel_std_list[i], self.kernel_std_list[i])))

        return kernels

    def forward(self, x, i, og=None):
        return self.kernels[i](x)


class DeColorization(ForwardProcessBase):

    def __init__(self, 
                 decolor_routine='Constant', 
                 decolor_ema_factor=0.9,
                 decolor_total_remove=False,
                 num_timesteps=50,
                 channels=3,
                 to_lab=False):

        self.decolor_routine = decolor_routine
        self.decolor_ema_factor = decolor_ema_factor
        self.decolor_total_remove = decolor_total_remove
        self.channels = channels
        self.num_timesteps = num_timesteps
        self.device_of_kernel = 'cuda'
        self.kernels = self.get_kernels()
        self.to_lab = to_lab

    def get_conv(self, decolor_ema_factor):
        conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, padding=0, padding_mode='circular',
                         bias=False)
        with torch.no_grad():
            ori_color_weight = torch.eye(self.channels)[:, :, None, None]
            decolor_weight = torch.ones((self.channels, self.channels)) / float(self.channels)
            decolor_weight = decolor_weight[:, :, None, None]
            kernel = decolor_ema_factor * ori_color_weight + (1.0 - decolor_ema_factor) * decolor_weight
            conv.weight = nn.Parameter(kernel)

        if self.device_of_kernel == 'cuda':
            conv = conv.cuda()

        return conv

    def get_kernels(self):
        kernels = []

        if self.decolor_routine == 'Constant':
            for i in range(self.num_timesteps):
                if i == self.num_timesteps - 1 and self.decolor_total_remove:
                    kernels.append(self.get_conv(0.0)) 
                else:
                    kernels.append(self.get_conv(self.decolor_ema_factor))
        elif self.decolor_routine == 'Linear':
            diff = 1.0 / self.num_timesteps
            start = 1.0
            for i in range(self.num_timesteps):
                if i == self.num_timesteps - 1 and self.decolor_total_remove:
                    kernels.append(self.get_conv(0.0)) 
                else:
                    # start * (1 - ema_factor) = diff
                    # ema_factor = 1 - diff / start
                    ema_factor = 1 - diff / start
                    start = start * ema_factor
                    kernels.append(self.get_conv(ema_factor))

        return kernels

    def forward(self, x, i, og=None):
        if self.to_lab:
            x_rgb = lab2rgb(x)
            x_next = self.kernels[i](x_rgb)
            return rgb2lab(x_next)
        else:
            return self.kernels[i](x)

    
    def total_forward(self, x_in):
        conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, padding=0, padding_mode='circular',
                         bias=False)
        if self.to_lab:
            x = lab2rgb(x_in)
        else:
            x = x_in

        with torch.no_grad():
            decolor_weight = torch.ones((self.channels, self.channels)) / float(self.channels)
            decolor_weight = decolor_weight[:, :, None, None]
            kernel = decolor_weight
            conv.weight = nn.Parameter(kernel)

        if self.device_of_kernel == 'cuda':
            conv = conv.cuda()

        x_out = conv(x)
        if self.to_lab:
            x_out = rgb2lab(x_out)
        return x_out

class Snow(ForwardProcessBase):
    
    def __init__(self,
                 image_size=(32,32),
                 snow_level=1,
                 num_timesteps=50,
                 snow_base_path=None,
                 random_snow=False,
                 single_snow=False,
                 batch_size=32,
                 load_snow_base=False,
                 fix_brightness=False):
        
        self.num_timesteps = num_timesteps
        self.random_snow = random_snow
        self.snow_level = snow_level
        self.image_size = image_size
        self.single_snow = single_snow
        self.batch_size = batch_size
        self.generate_snow_layer()
        self.fix_brightness = fix_brightness
    
    @torch.no_grad()
    def reset_parameters(self, batch_size=-1):
        if batch_size != -1:
            self.batch_size = batch_size
        if self.random_snow:
            self.generate_snow_layer()



    @torch.no_grad()
    def generate_snow_layer(self):
        if not self.random_snow:
            rstate = np.random.get_state()
            np.random.seed(123321)
        # c[0]/c[1]: mean/std of Gaussian for snowy pixels
        # c[2]: zoom factor
        # c[3]: threshold for snowy pixels
        # c[4]/c[5]: radius/sigma for motion blur
        # c[6]: brightness coefficient
        if self.snow_level == 1:
            c = (0.1, 0.3, 3, 0.5, 5, 4, 0.8)
            snow_thres_start = 0.7
            snow_thres_end = 0.3
            mb_sigma_start = 0.5
            mb_sigma_end = 5.0
            br_coef_start = 0.95
            br_coef_end = 0.7
        elif self.snow_level == 2:
            c = (0.55, 0.3, 2.5, 0.85, 11, 12, 0.55) 
            snow_thres_start = 1.15
            snow_thres_end = 0.7
            mb_sigma_start = 0.05
            mb_sigma_end = 12
            br_coef_start = 0.95
            br_coef_end = 0.55
        elif self.snow_level == 3:
            c = (0.55, 0.3, 2.5, 0.7, 11, 16, 0.4) 
            snow_thres_start = 1.15
            snow_thres_end = 0.7
            mb_sigma_start = 0.05
            mb_sigma_end = 16
            br_coef_start = 0.95
            br_coef_end = 0.4
        elif self.snow_level == 4:
            c = (0.55, 0.3, 2.5, 0.55, 11, 20, 0.3) 
            snow_thres_start = 1.15
            snow_thres_end = 0.55
            mb_sigma_start = 0.05
            mb_sigma_end = 20
            br_coef_start = 0.95
            br_coef_end = 0.3



        self.snow_thres_list = torch.linspace(snow_thres_start, snow_thres_end, self.num_timesteps).tolist()

        self.mb_sigma_list = torch.linspace(mb_sigma_start, mb_sigma_end, self.num_timesteps).tolist()

        self.br_coef_list = torch.linspace(br_coef_start, br_coef_end, self.num_timesteps).tolist()


        self.snow = []
        self.snow_rot = []
        
        if self.single_snow:
            sb_list = []
            for _ in range(self.batch_size):
                cs = np.random.normal(size=self.image_size, loc=c[0], scale=c[1])
                cs = cs[..., np.newaxis]
                cs = clipped_zoom(cs, c[2])
                sb_list.append(cs)
            snow_layer_base = np.concatenate(sb_list, axis=2)
        else:
            snow_layer_base = np.random.normal(size=self.image_size, loc=c[0], scale=c[1])
            snow_layer_base = snow_layer_base[..., np.newaxis]
            snow_layer_base = clipped_zoom(snow_layer_base, c[2])
        
        vertical_snow = False
        if np.random.uniform() > 0.5:
            vertical_snow = True

        for i in range(self.num_timesteps):

            snow_layer = torch.Tensor(snow_layer_base).clone()
            snow_layer[snow_layer < self.snow_thres_list[i]] = 0
            snow_layer = torch.clip(snow_layer, 0, 1)
            snow_layer = snow_layer.permute((2, 0, 1)).unsqueeze(1)
            # Apply motion blur
            kernel_param = tgm.image.get_gaussian_kernel(c[4], self.mb_sigma_list[i])
            motion_kernel = torch.zeros((c[4], c[4]))
            motion_kernel[int(c[4] / 2)] = kernel_param

            horizontal_kernel = motion_kernel[None, None, :]
            horizontal_kernel = horizontal_kernel.repeat(3, 1, 1, 1)
            vertical_kernel = torch.rot90(motion_kernel, k=1, dims=[0,1])
            vertical_kernel = vertical_kernel[None, None, :]
            vertical_kernel = vertical_kernel.repeat(3, 1, 1, 1)

            vsnow = F.conv2d(snow_layer, vertical_kernel, padding='same', groups=1)
            hsnow = F.conv2d(snow_layer, horizontal_kernel, padding='same', groups=1)
            if self.single_snow:
                vidx = torch.randperm(snow_layer.shape[0])
                vidx = vidx[:int(snow_layer.shape[0]/2)]
                snow_layer = hsnow
                snow_layer[vidx] = vsnow[vidx]
            elif vertical_snow:
                snow_layer = vsnow
            else:
                snow_layer = hsnow
            self.snow.append(snow_layer)
            self.snow_rot.append(torch.rot90(snow_layer, k=2, dims=[2,3]))
        
        if not self.random_snow:
            np.random.set_state(rstate)

    @torch.no_grad()
    def total_forward(self, x_in):
        return self.forward(None, self.num_timesteps-1, og=x_in)
    
    @torch.no_grad()
    def forward(self, x, i, og=None):
        og_r = (og + 1.) / 2.
        og_gray = rgb_to_grayscale(og_r) * 1.5 + 0.5
        og_gray = torch.maximum(og_r, og_gray)
        br_coef = self.br_coef_list[i]
        scaled_og = br_coef * og_r + (1 - br_coef) * og_gray
        if self.fix_brightness:
            snowy_img = torch.clip(og_r + self.snow[i].cuda() + self.snow_rot[i].cuda(), 0.0, 1.0)
        else:
            snowy_img = torch.clip(scaled_og + self.snow[i].cuda() + self.snow_rot[i].cuda(), 0.0, 1.0)
        return (snowy_img * 2.) - 1.
