from comet_ml import Experiment
from deblurring_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, Model
import torchvision
import os
import errno
import shutil
import argparse

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

create = 0

if create:
    trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True)
    root = './root_cifar10/'
    del_folder(root)
    create_folder(root)

    for i in range(10):
        lable_root = root + str(i) + '/'
        create_folder(lable_root)

    for idx in range(len(trainset)):
        img, label = trainset[idx]
        print(idx)
        img.save(root + str(label) + '/' + str(idx) + '.png')


parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int,
                    help="This is the number of steps in which a clean image looses information.")
parser.add_argument('--train_steps', default=700000, type=int,
                    help='The number of iterations for training.')
parser.add_argument('--blur_std', default=0.1, type=float,
                    help='It sets the standard deviation for blur routines which have different meaning based on blur routine.')
parser.add_argument('--blur_size', default=3, type=int,
                    help='It sets the size of gaussian blur used in blur routines for each step t')
parser.add_argument('--image_size', default=32, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--save_folder', default='./results_cifar10', type=str)
parser.add_argument('--data_path', default='./root_cifar10/', type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--blur_routine', default='Incremental', type=str,
                    help='This will set the type of blur routine one can use, check the code for what each one of them does in detail')
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str,
                    help='The choice of sampling routine for reversing the diffusion process, when set as default it corresponds to Alg. 1 while when set as x0_step_down it stands for Alg. 2')
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--discrete', action="store_true")
parser.add_argument('--da', default='cifar10', type=str)


args = parser.parse_args()
print(args)


model = Model(resolution=args.image_size,
              in_channels=3,
              out_ch=3,
              ch=128,
              ch_mult=(1,2,2,2),
              num_res_blocks=2,
              attn_resolutions=(16,),
              dropout=0.1).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = args.image_size,
    device_of_kernel = 'cuda',
    channels = 3,
    timesteps = args.time_steps,   # number of steps
    loss_type = 'l1',    # L1 or L2
    kernel_std=args.blur_std,
    kernel_size=args.blur_size,
    blur_routine=args.blur_routine,
    train_routine = args.train_routine,
    sampling_routine = args.sampling_routine,
    discrete=args.discrete
).cuda()

import torch
diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))


trainer = Trainer(
    diffusion,
    args.data_path,
    image_size = args.image_size,
    train_batch_size = args.batch_size,
    train_lr = 2e-5,
    train_num_steps = args.train_steps,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
    dataset = args.da
)

trainer.train()