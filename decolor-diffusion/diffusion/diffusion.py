import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import time


from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from torchvision import datasets
from PIL import Image

import numpy as np
from tqdm import tqdm
from einops import rearrange

import pickle

import torchgeometry as tgm
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch import linalg as LA
import imageio
from .forward_process_impl import DeColorization, Snow
from .get_dataset import get_dataset
from .utils import rgb2lab, lab2rgb

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

def create_folder(path):
    path_to_create = Path(path)
    path_to_create.mkdir(parents=True, exist_ok = True)

def cycle(dl, f=None):
    while True:
        for data in dl:
            # Temporary fix for torchvision CIFAR-10
            if type(data) == list:
                yield f(data[0])
            else:
                yield f(data)

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
        one_shot_denoise_fn = None,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        kernel_std = 0.1,
        kernel_size = 3,
        forward_process_type = 'Decolorization',
        train_routine = 'Final',
        sampling_routine='default',
        start_kernel_std=0.01,
        target_kernel_std=1.0,
        decolor_routine='Constant',
        decolor_ema_factor=0.9,
        decolor_total_remove=True,
        snow_level=1,
        random_snow=False,
        to_lab=False,
        order_seed=-1.0,
        recon_noise_std=0.0,
        load_snow_base=False,
        load_path=None,
        batch_size=32,
        single_snow=False,
        fix_brightness=False,
        results_folder=None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.device_of_kernel = device_of_kernel
        

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.train_routine = train_routine
        self.sampling_routine = sampling_routine



        self.snow_level = snow_level
        self.random_snow = random_snow
        self.batch_size = batch_size
        self.single_snow = single_snow

        self.to_lab = to_lab
        self.recon_noise_std = recon_noise_std
        # mixing coef of loss of pred vs ground_truth 

        # Gaussian Blur parameters
                                    
        if forward_process_type == 'Decolorization':
            self.forward_process = DeColorization(decolor_routine=decolor_routine,
                                                  decolor_ema_factor=decolor_ema_factor,
                                                  decolor_total_remove=decolor_total_remove,
                                                  channels=self.channels,
                                                  num_timesteps=self.num_timesteps,
                                                  to_lab=self.to_lab,)
        elif forward_process_type == 'Snow':
            if load_path is not None:
                snow_base_path = load_path.replace('model.pt', 'snow_base.npy')
                print(snow_base_path)
                load_snow_base = True
            else:
                snow_base_path = os.path.join(results_folder, 'snow_base.npy')
                load_snow_base = False
            self.forward_process = Snow(image_size=self.image_size,
                                        snow_level=self.snow_level, 
                                        random_snow=self.random_snow,
                                        num_timesteps=self.num_timesteps,
                                        snow_base_path=snow_base_path,
                                        batch_size=self.batch_size,
                                        single_snow=self.single_snow,
                                        load_snow_base=load_snow_base,
                                        fix_brightness=fix_brightness)
    
    @torch.no_grad()
    def sample_one_step(self, img, t, init_pred=None):
        orig_mean = torch.mean(img, [2, 3], keepdim=True)

        #x = self.denoise_fn(img, t)
        x = self.prediction_step_t(img, t, init_pred)
        direct_recons = x.clone()
        
        if self.recon_noise_std > 0.0:
            self.recon_noise_std_array = torch.linspace(0.0, self.recon_noise_std, steps=self.num_timesteps)

        if self.train_routine in ['Final', 'Final_random_mean', 'Final_small_noise', 'Final_random_mean_and_actual']:

            if self.sampling_routine == 'default':

                x_times_sub_1 = x.clone()
                cur_time = torch.zeros_like(t)
                fp_index = torch.where(cur_time < t - 1)[0]
                for i in range(t.max() - 1):
                    x_times_sub_1[fp_index] = self.forward_process.forward(x_times_sub_1[fp_index], i, og=x[fp_index])
                    cur_time += 1
                    fp_index = torch.where(cur_time < t - 1)[0]

                x = x_times_sub_1


            elif self.sampling_routine == 'x0_step_down':
                
                x_times = x.clone()
                if self.recon_noise_std > 0.0:
                    x_times = x + torch.normal(0.0, self.recon_noise_std, size=x.size()).cuda()
                x_times_sub_1 = x_times.clone()

                cur_time = torch.zeros_like(t)
                fp_index = torch.where(cur_time < t)[0]
                for i in range(t.max()):
                    x_times_sub_1 = x_times.clone()
                    x_times[fp_index] = self.forward_process.forward(x_times[fp_index], i, og=x[fp_index])
                    cur_time += 1
                    fp_index = torch.where(cur_time < t)[0]


                x = img - x_times + x_times_sub_1

        elif self.train_routine == 'Step':
            img = x

        elif self.train_routine == 'Step_Gradient':
            x = img + x
        
        return x, direct_recons

    @torch.no_grad()
    def sample_multi_step(self, img, t_start, t_end):
        fp_index = torch.where(t_start > t_end)[0]
        img_new= img.clone()
        while len(fp_index) > 0:
            _, img_new_partial = self.sample_one_step(img_new[fp_index], t_start[fp_index])
            img_new[fp_index] = img_new_partial
            t_start = t_start - 1
            fp_index = torch.where(t_start > t_end)[0]
        return img_new


    @torch.no_grad()
    def sample(self, batch_size = 16, img=None, t=None):

        self.forward_process.reset_parameters(batch_size=batch_size)
        if t==None:
            t=self.num_timesteps

        
        degrade_dict = {}
        og_img = img.clone()
        
        for i in range(t):
            print(i)
            with torch.no_grad():
                img = self.forward_process.forward(img, i, og=og_img)
        
        init_pred = None
        # 3(2), 2(1), 1(0)
        xt = img
        direct_recons = None
        while(t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x, cur_direct_recons = self.sample_one_step(img, step, init_pred=init_pred)
            if direct_recons is None:
                direct_recons = cur_direct_recons
            img = x
            t = t - 1
        
        if self.to_lab:
            xt = lab2rgb(xt)
            direct_recons = lab2rgb(direct_recons)
            img = lab2rgb(img)
            
        return_dict =  {'xt': xt, 
                        'direct_recons': direct_recons, 
                        'recon': img,}
        return return_dict


    @torch.no_grad()
    def all_sample(self, batch_size=16, img=None, t=None, times=None, res_dict=None):
        
        self.forward_process.reset_parameters(batch_size=batch_size)
        if t == None:
            t = self.num_timesteps
        if times == None:
            times = t
        
        
        img_forward_list = []
        
        img_forward = img

        with torch.no_grad():
            img = self.forward_process.total_forward(img)

    

        X_0s = []
        X_ts = []

        init_pred = None
        while (times):
            step = torch.full((img.shape[0],), times - 1, dtype=torch.long).cuda()
            img, direct_recons = self.sample_one_step(img, step, init_pred=init_pred)
            X_0s.append(direct_recons.cpu())
            X_ts.append(img.cpu())
            times = times - 1

        init_pred_clone = None
        if init_pred is not None:
            init_pred_clone = init_pred.clone().cpu()
        if self.to_lab:
            for i in range(len(X_0s)):
                X_0s[i] = lab2rgb(X_0s[i])
                X_ts[i] = lab2rgb(X_ts[i])
            if init_pred is not None:
                init_pred_clone = lab2rgb(init_pred_clone)


        return X_0s, X_ts, init_pred_clone, img_forward_list




    def q_sample(self, x_start, t, return_total_blur=False):
        # So at present we will for each batch blur it till the max in t.
        # And save it. And then use t to pull what I need. It is nothing but series of convolutions anyway.
        # Remember to do convs without torch.grad
        
        final_sample = x_start.clone()
        
        noisy_index = torch.where(t == -1)[0]


        max_iters = torch.max(t)
        all_blurs = []
        x = x_start[torch.where(t != -1)]
        blurring_batch_size = x.shape[0]
        if blurring_batch_size == 0:
            return final_sample

        for i in range(max_iters+1):
            with torch.no_grad():
                x = self.forward_process.forward(x, i, og=final_sample[torch.where(t != -1)])
                all_blurs.append(x)

                if i == max_iters:
                    total_blur = x.clone()

        all_blurs = torch.stack(all_blurs)

        choose_blur = []
        # step is batch size as well so for the 49th step take the step(batch_size)
        for step in range(blurring_batch_size):
            if step != -1:
                choose_blur.append(all_blurs[t[step], step])
            else:
                choose_blur.append(x_start[step])

        choose_blur = torch.stack(choose_blur)
        #choose_blur = all_blurs

        final_sample[torch.where(t != -1)] = choose_blur

        if return_total_blur:
            final_sample_total_blur = final_sample.clone()
            final_sample_total_blur[torch.where(t != -1)] = total_blur
            return final_sample, final_sample_total_blur
        return final_sample
    
    def loss_func(self, pred, true):
        if self.loss_type == 'l1':
            return (pred - true).abs().mean()
        elif self.loss_type == 'l2':
            return F.mse_loss(pred, true)
        elif self.loss_type == 'sqrt':
            return (pred - true).abs().mean().sqrt()
        else:
            raise NotImplementedError()

    
    
    def prediction_step_t(self, img, t, init_pred=None):
        return self.denoise_fn(img, t)

    def p_losses(self, x_start, t, t_pred=None):
        b, c, h, w = x_start.shape
        
        self.forward_process.reset_parameters()

        if self.train_routine == 'Final':
            x_blur, x_total_blur = self.q_sample(x_start=x_start, t=t, return_total_blur=True)

            x_recon = self.denoise_fn(x_blur, t)
            loss = self.loss_func(x_start, x_recon)
            
        elif self.train_routine == 'Step_Gradient':
            x_blur, x_total_blur = self.q_sample(x_start=x_start, t=t, return_total_blur=True)
            x_blur_sub = self.q_sample(x_start=x_start, t=t-1)

            x_blur_diff = x_blur_sub - x_blur
            x_blur_diff_pred = self.denoise_fn(x_blur, t)
            loss = self.loss_func(x_blur_diff, x_blur_diff_pred)

        elif self.train_routine == 'Step':
            x_blur, x_total_blur = self.q_sample(x_start=x_start, t=t, return_total_blur=True)
            x_blur_sub = self.q_sample(x_start=x_start, t=t-1)
            
            x_blur_sub_pred = self.denoise_fn(x_blur, t)
            loss = self.loss_func(x_blur_sub, x_blur_sub_pred)

        return loss

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        if type(img_size) is tuple:
            img_w, img_h = img_size
        else:
            img_h, img_w = img_size, img_size
        assert h == img_h and w == img_w, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        t_pred = [] 
        for i in range(b):
            t_pred.append(torch.randint(0, t[i]+1, ()).item())
        t_pred = torch.Tensor(t_pred).to(device).long()  -1
        t_pred[t_pred < 0] = 0

        return self.p_losses(x, t, t_pred, *args, **kwargs)

    @torch.no_grad()
    def forward_and_backward(self, batch_size=16, img=None, t=None, times=None, eval=True):

        self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps



        Forward = []
        Forward.append(img)


        for i in range(t):
            with torch.no_grad():
                step = torch.full((batch_size,), i, dtype=torch.long, device=img.device)
                n_img = self.q_sample(x_start=img, t=step)
                Forward.append(n_img)

        Backward = []
        img = n_img
        while (t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
            x1_bar = self.denoise_fn(img, step)

            Backward.append(img)

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, t=step2)

            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1

        return Forward, Backward, img
# dataset classes

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png'], random_aug=False):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = Dataset.get_transform(self.image_size, random_aug=random_aug)

    def get_transform(image_size, random_aug=False):
        if image_size[0] == 256:
            T = transforms.Compose([
                transforms.CenterCrop((128,128)),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
            ])
        elif not random_aug:
            T = transforms.Compose([
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
            ])
        else:
            s = 1.0
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            T = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1)
            ])

        return T

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)



class Dataset_Cifar10(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
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
        save_and_sample_every = 5000,
        save_with_time_stamp_every = 50000,
        results_folder = './results',
        load_path = None,
        random_aug=False,
        torchvision_dataset=False,
        dataset = None,
        to_lab=False,
        order_seed=-1,
    ):
        super().__init__()
        self.model = diffusion_model
        self.num_timesteps = diffusion_model.num_timesteps
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.save_with_time_stamp_every = save_with_time_stamp_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
    

        self.to_lab = to_lab
        self.order_seed = order_seed

        self.random_aug = random_aug
        if torchvision_dataset:
            self.ds = get_dataset(dataset, folder, self.image_size, random_aug=random_aug)
        else:
            self.ds = Dataset(folder, image_size, random_aug=self.random_aug)
        post_process_func = lambda x: x
        if self.to_lab:
            post_process_func = rgb2lab
        
        self.order_seed = int(self.order_seed)
        if self.order_seed == -1:
            self.data_loader = data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True, num_workers=4)
        else:
            self.data_loader = data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True, num_workers=4)

        
        self.post_process_func = post_process_func
        self.dl = cycle(self.data_loader, f=post_process_func)

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0


        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok = True)

        self.reset_parameters()

        if load_path != None:
            self.load(load_path)
    
    def _process_item(self, x):
        f = self.post_process_func
        if type(x) == list:
            return f(x[0])
        else:
            return f(x)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, save_with_time_stamp=False):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        if save_with_time_stamp:
            torch.save(data, str(self.results_folder / f'model_{self.step}.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data['step']
        print("Model at step : ", self.step)
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def add_title(self, path, title_texts):
        import cv2
        import numpy as np

        img1 = cv2.imread(path)

        # --- Here I am creating the border---
        black = [0, 0, 0]  # ---Color of the border---
        constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        #cv2.imshow('constant', constant)
        height = 20
        violet = np.zeros((height, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)

        vcat = cv2.vconcat((violet, constant))
        #cv2.imshow('vcat', vcat)

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        title_counts = len(title_texts)
        for i, title in enumerate(title_texts):
            vertical_pos = i * (violet.shape[1] // title_counts) + (violet.shape[1] // (title_counts * 2))
            cv2.putText(vcat, str(title), (vertical_pos, height-2), font, 0.5, (0, 0, 0), 1, 0)
        cv2.imwrite(path, vcat)
 
    def make_transparent(self, path):
        img = mpimg.imread(path)
        plt.imshow(img)
        plt.savefig(path, transparent=True)
        plt.close()
    
    def train(self):
        backwards = partial(loss_backwards, self.fp16)
        
        start_time = time.time()
        

        while self.step < self.train_num_steps:

            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()
                loss = self.model(data)
                print(f'{self.step}: {loss.item()}')
                backwards(loss / self.gradient_accumulate_every, self.opt)

            if self.step != 0 and self.step % 100 == 0:
                print(f'time for 100 steps: {time.time() - start_time}')
                start_time = time.time()

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                #if 'flower' not in args.dataset or 'cifar':
                if True:
                    milestone = self.step // self.save_and_sample_every
                    batches = self.batch_size
                    og_img = next(self.dl).cuda()
                    sample_dict = self.ema_model.sample(batch_size=batches, img=og_img)
                    if self.to_lab:
                        og_img = lab2rgb(og_img)
                    sample_dict['og'] = og_img

                    print(f'images saved: {sample_dict.keys()}')
                    for k, img in sample_dict.items():
                        img_scale = (img + 1) * 0.5
                        utils.save_image(img_scale, str(self.results_folder / f'sample-{k}-{milestone}.png'), nrow=6)

                self.save()
            # Save model with time stamp
            if self.step != 0 and self.step % self.save_with_time_stamp_every == 0:
                self.save(save_with_time_stamp=True)


            self.step += 1

        print('training completed')
    

    def save_gif(self, X_0s, X_ts, extra_path, init_recon=None, og=None):

        frames_t = []
        frames_0 = []
        to_PIL = transforms.ToPILImage()
        
        if init_recon is not None:
            init_recon = (init_recon + 1) * 0.5
        self.gif_len = len(X_0s)
        for i in range(len(X_0s)):

            print(i)
            start_time = time.time()
            x_0 = X_0s[i]
            x_0 = (x_0 + 1) * 0.5
            x_0_grid = utils.make_grid(x_0, nrow=6)
            if init_recon is not None:
                init_recon_grid = utils.make_grid(init_recon, nrow=6)
                x_0_grid = utils.make_grid(torch.stack((x_0_grid, og, init_recon_grid)), nrow=3)
                title_texts = [str(i), 'og', 'init_recon']
            elif og is not None:
                x_0_grid = utils.make_grid(torch.stack((x_0_grid, og)), nrow=2)
                title_texts = [str(i), 'og']
            utils.save_image(x_0_grid, str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'))
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), title_texts)
            frames_0.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png')))


            x_t = X_ts[i]
            all_images = (x_t + 1) * 0.5
            all_images_grid = utils.make_grid(all_images, nrow=6)

            if init_recon is not None:
                init_recon_grid = utils.make_grid(init_recon, nrow=6)
                all_images_grid = utils.make_grid(torch.stack((all_images_grid, og, init_recon_grid)), nrow=3)
            elif og is not None:
                all_images_grid = utils.make_grid(torch.stack((all_images_grid, og)), nrow=2)
            utils.save_image(all_images_grid, str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'))
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), title_texts)
            frames_t.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png')))


        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), frames_0)
        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), frames_t)

    
    def save_og_test(self, og_dict, extra_path):

        for k, img in og_dict.items():
            img_scale = (img + 1) * 0.5
            img_grid = utils.make_grid(img_scale, nrow=6)
            utils.save_image(img_grid, str(self.results_folder / f'{k}-{extra_path}.png'))
            self.add_title(str(self.results_folder / f'{k}-{extra_path}.png'), '{k}')
            og_dict[k] = img_grid
    
    def create_metric_dict(self):
        step_metrics_FID = [metrics.FID() for _ in range(self.num_timesteps)]
        step_metrics_PSNR = [metrics.PSNR(data_range=1.0) for _ in range(self.num_timesteps)]
        step_metrics_SSIM = [metrics.SSIM(data_range=1.0) for _ in range(self.num_timesteps)]

        #return {'FID': step_metrics_FID, 'PSNR': step_metrics_PSNR, 'SSIM': step_metrics_SSIM}
        return {'PSNR': step_metrics_PSNR, 'SSIM': step_metrics_SSIM}
    
    def save_metric(self, metric_dict, prefix=''):
        for k, v in metric_dict.items():
            save_file_name = str(self.results_folder / f'{prefix}-{k}.txt')
            with open(save_file_name, 'w') as f:
                score_list = [f'{str(m.compute().item())}\n' for m in v]
                f.writelines(score_list)
        
    
    def shift_data_range(self, img):
        return (img + 1.0) / 2

    def test_from_data(self, extra_path, s_times=None):
        batches = self.batch_size
        #og_img = next(self.dl).cuda()
        
        #xt_metric_dict = self.create_metric_dict()
        #forward_metric_dict = self.create_metric_dict()

        for batch_idx, x in enumerate(self.data_loader): 
            og_img = self._process_item(x).cuda()
            og_dict = {'og': og_img.cuda()}
            X_0s, X_ts, init_recon, img_forward_list = self.ema_model.all_sample(batch_size=batches, img=og_img, times=s_times, res_dict=og_dict)
            og_dict['og'] = og_img.cpu()
            print(f'Generating on batch {batch_idx}')
            if batch_idx == 0:
                self.save_og_test(og_dict, extra_path)
                self.save_gif(X_0s, X_ts, extra_path, init_recon=init_recon, og=og_dict['og'])

                return

            if batch_idx * batches > 1000:
                break
            if og_img.shape[0] != batches:
                continue
            assert len(X_ts) == len(img_forward_list)

        print(f'Finish sample generation')

    def test_with_mixup(self, extra_path):
        batches = self.batch_size
        og_img_1 = next(self.dl).cuda()
        og_img_2 = next(self.dl).cuda()
        og_img = (og_img_1 + og_img_2)/2

        X_0s, X_ts = self.ema_model.all_sample(batch_size=batches, img=og_img)
        print(f'Finish sample generation')

        og_dict = {
                  'og1': og_img_1,
                  'og2': og_img_2,
                  'og': og_img,
        }
        
        self.save_og_test(og_dict, extra_path)
        self.save_gif(X_0s, X_ts, extra_path, og=og_dict['og'])


    def test_from_random(self, extra_path):
        batches = self.batch_size
        og_img = next(self.dl).cuda()
        og_img = og_img*0.9 #torch.randn_like(og_img) + 0.1
        og_dict = {'og': og_img}

        X_0s, X_ts = self.ema_model.all_sample(batch_size=batches, img=og_img, res_dict=og_dict)
        print(f'Finish sample generation')
        
        
        self.save_og_test(og_dict, extra_path)
        self.save_gif(X_0s, X_ts, extra_path, og=og_dict['og'])

    def paper_invert_section_images(self, s_times=None):

        cnt = 0
        max_pixel_diff = 0.0 
        batch_idx = -1 
        max_pixel_diff_sum = -1.0
        batch_idx_sum_diff = -1
        for i in range(20):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            X_0s, X_ts, _, _ = self.ema_model.all_sample(batch_size=batches, img=og_img, times=s_times)
            og_img = (og_img + 1) * 0.5
            


            for j in range(og_img.shape[0]//9):
                original = og_img[j: j + 9]
                utils.save_image(original, str(self.results_folder / f'original_{cnt}.png'), nrow=3)

                direct_recons = X_0s[0][j: j + 9]
                direct_recons = (direct_recons + 1) * 0.5
                utils.save_image(direct_recons, str(self.results_folder / f'direct_recons_{cnt}.png'), nrow=3)

                sampling_recons = X_0s[-1][j: j + 9]
                sampling_recons = (sampling_recons + 1) * 0.5
                utils.save_image(sampling_recons, str(self.results_folder / f'sampling_recons_{cnt}.png'), nrow=3)

                diff = (direct_recons - sampling_recons).squeeze().sum(dim=0).max()
                diff_sum = (direct_recons - sampling_recons).squeeze().sum(dim=0).sum()
                if diff > max_pixel_diff:
                    batch_idx = j
                    max_pixel_diff = diff

                if diff_sum > max_pixel_diff_sum:
                    batch_idx_sum_diff = j
                    max_pixel_diff_sum = diff_sum

                blurry_image = X_ts[0][j: j + 9]
                blurry_image = (blurry_image + 1) * 0.5
                utils.save_image(blurry_image, str(self.results_folder / f'blurry_image_{cnt}.png'), nrow=3)



                import cv2

                blurry_image = cv2.imread(f'{self.results_folder}/blurry_image_{cnt}.png')
                direct_recons = cv2.imread(f'{self.results_folder}/direct_recons_{cnt}.png')
                sampling_recons = cv2.imread(f'{self.results_folder}/sampling_recons_{cnt}.png')
                original = cv2.imread(f'{self.results_folder}/original_{cnt}.png')

                #black = [255, 255, 255]
                black = [0, 0, 0]
                blurry_image = cv2.copyMakeBorder(blurry_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                direct_recons = cv2.copyMakeBorder(direct_recons, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                sampling_recons = cv2.copyMakeBorder(sampling_recons, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                original = cv2.copyMakeBorder(original, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)

                im_h = cv2.hconcat([blurry_image, direct_recons, sampling_recons, original])
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)

                cnt += 1 


    def paper_showing_diffusion_images(self, s_times=None):

        import cv2
        cnt = 0
        to_show = [0, 1, 2, 4, 8, 16, 24, 32, 40, 44, 46, 48, 49]

        for i in range(5):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            X_0s, X_ts, _, _ = self.ema_model.all_sample(batch_size=batches, img=og_img, times=s_times)
            og_img = (og_img + 1) * 0.5

            for k in range(X_ts[0].shape[0]):
                l = []

                for j in range(len(X_ts)):
                    x_t = X_ts[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(x_t, str(self.results_folder / f'x_{len(X_ts)-j}_{cnt}.png'), nrow=1)
                    x_t = cv2.imread(f'{self.results_folder}/x_{len(X_ts)-j}_{cnt}.png')
                    if j in to_show:
                        l.append(x_t)


                x_0 = X_0s[-1][k]
                x_0 = (x_0 + 1) * 0.5
                utils.save_image(x_0, str(self.results_folder / f'x_best_{cnt}.png'), nrow=1)
                x_0 = cv2.imread(f'{self.results_folder}/x_best_{cnt}.png')
                l.append(x_0)
                im_h = cv2.hconcat(l)
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)

                cnt+=1



    def fid_distance_decrease_from_manifold(self, fid_func, start=0, end=1000):
        #from skimage.metrics import structural_similarity as ssim
        from pytorch_msssim import ssim

        all_samples = []
        dataset = self.ds

        print(len(dataset))
        perp = np.random.permutation(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[perp[idx]]
            if type(img) is tuple:
                img = img[0]
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
        # create_folder(f'{self.results_folder}/')
        blurred_samples = None
        original_sample = None
        deblurred_samples = None
        direct_deblurred_samples = None

        sanity_check = 0




        cnt=0
        eval_batch_size = 16
        while(cnt < all_samples.shape[0]):
            og_x = all_samples[cnt: cnt + eval_batch_size]
            og_x = og_x.cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            og_img = og_x
            print(og_img.shape)
            X_0s, X_ts, _, _ = self.ema_model.all_sample(batch_size=og_img.shape[0], img=og_img, times=None)

            og_img = og_img.to('cpu')
            blurry_imgs = X_ts[0].to('cpu')
            deblurry_imgs = X_0s[-1].to('cpu')
            direct_deblurry_imgs = X_0s[0].to('cpu')

            if og_img.shape[2] > 256:
                og_img = F.interpolate(og_img, size=64)
                blurry_imgs = F.interpolate(blurry_imgs, size=64)
                deblurry_imgs = F.interpolate(deblurry_imgs, size=64)
                direct_deblurry_imgs = F.interpolate(direct_deblurry_imgs, size=64)

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

    def paper_showing_diffusion_images_cover_page(self):

        import cv2
        cnt = 0
        to_show = [int(self.num_timesteps * i / 4) for i in range(4)]
        to_show.append(self.num_timesteps - 1)

        for i in range(50):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            Forward, Backward, final_all = self.ema_model.forward_and_backward(batch_size=batches, img=og_img)
            og_img = (og_img + 1) * 0.5
            final_all = (final_all + 1) * 0.5



            for k in range(Forward[0].shape[0]):
                l = []

                utils.save_image(og_img[k], str(self.results_folder / f'og_img_{cnt}.png'), nrow=1)
                start = cv2.imread(f'{self.results_folder}/og_img_{cnt}.png')
                l.append(start)

                for j in range(len(Forward)):
                    x_t = Forward[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(x_t, str(self.results_folder / f'temp.png'), nrow=1)
                    x_t = cv2.imread(f'{self.results_folder}/temp.png')
                    if j in to_show:
                        l.append(x_t)

                for j in range(len(Backward)):
                    x_t = Backward[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(x_t, str(self.results_folder / f'temp.png'), nrow=1)
                    x_t = cv2.imread(f'{self.results_folder}/temp.png')
                    if (len(Backward) - j) in to_show:
                        l.append(x_t)


                utils.save_image(final_all[k], str(self.results_folder / f'final_{cnt}.png'), nrow=1)
                final = cv2.imread(f'{self.results_folder}/final_{cnt}.png')
                l.append(final)


                im_h = cv2.hconcat(l)
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)

                cnt+=1 


