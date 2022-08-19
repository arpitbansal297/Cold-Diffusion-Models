from resolution_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torchvision
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--train_steps', default=700000, type=int)
parser.add_argument('--save_folder', default='./results_celebA', type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--data_path', default='./root_celebA_128_train_new/', type=str)
parser.add_argument('--resolution_routine', default='Incremental', type=str)
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")

args = parser.parse_args()
print(args)

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=3,
    with_time_emb=not(args.remove_time_embed),
    residual=args.residual
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    device_of_kernel = 'cuda',
    channels = 3,
    timesteps = args.time_steps,        # number of steps
    loss_type = 'l1',                   # L1 or L2
    resolution_routine=args.resolution_routine,
    train_routine = args.train_routine,
    sampling_routine = args.sampling_routine
).cuda()

import torch
diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

trainer = Trainer(
    diffusion,
    args.data_path,
    image_size = 128,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = args.train_steps, # total training steps
    gradient_accumulate_every = 2,      # gradient accumulation steps
    ema_decay = 0.995,                  # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
    dataset = 'celebA'
)

trainer.train()
