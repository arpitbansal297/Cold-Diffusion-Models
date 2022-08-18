from defading_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torchvision
import os
import errno
import shutil
import argparse
from pathlib import Path
from PIL import Image


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


create = 1

if create:
    root_train = './root_celebA_128_train/'
    root_test = './root_celebA_128_test/'
    del_folder(root_train)
    create_folder(root_train)

    del_folder(root_test)
    create_folder(root_test)

    exts = ['jpg', 'jpeg', 'png']
    folder = '/fs/cml-datasets/CelebA-HQ/images-128/'
    paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

    for idx in range(len(paths)):
        img = Image.open(paths[idx])
        print(idx)
        if idx % 10 == 0:
            img.save(root_test + str(idx) + '.png')
        else:
            img.save(root_train + str(idx) + '.png')
#
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--time_steps', default=50, type=int)
# parser.add_argument('--train_steps', default=700000, type=int)
# parser.add_argument('--blur_std', default=0.1, type=float)
# parser.add_argument('--blur_size', default=3, type=int)
# parser.add_argument('--save_folder', default='./results_cifar10', type=str)
# parser.add_argument('--load_path', default=None, type=str)
# parser.add_argument('--blur_routine', default='Incremental', type=str)
# parser.add_argument('--train_routine', default='Final', type=str)
# parser.add_argument('--sampling_routine', default='default', type=str)
# parser.add_argument('--remove_time_embed', action="store_true")
# parser.add_argument('--residual', action="store_true")
#
#
#
# args = parser.parse_args()
# print(args)
#
#
# model = Unet(
#     dim = 64,
#     dim_mults = (1, 2, 4, 8),
#     channels=3,
#     with_time_emb=not(args.remove_time_embed),
#     residual=args.residual
# ).cuda()
#
# import torch
# model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
#
#
# diffusion = GaussianDiffusion(
#     model,
#     image_size = 128,
#     device_of_kernel = 'cuda',
#     channels = 3,
#     timesteps = args.time_steps,   # number of steps
#     loss_type = 'l1',    # L1 or L2
#     kernel_std=args.blur_std,
#     kernel_size=args.blur_size,
#     blur_routine=args.blur_routine,
#     train_routine = args.train_routine,
#     sampling_routine = args.sampling_routine
# ).cuda()
#
# trainer = Trainer(
#     diffusion,
#     './root_celebA_128_train/',
#     image_size = 128,
#     train_batch_size = 32,
#     train_lr = 2e-5,
#     train_num_steps = args.train_steps,         # total training steps
#     gradient_accumulate_every = 2,    # gradient accumulation steps
#     ema_decay = 0.995,                # exponential moving average decay
#     fp16 = False,                       # turn on mixed precision training with apex
#     results_folder = args.save_folder,
#     load_path = args.load_path
# )
#
# trainer.train()