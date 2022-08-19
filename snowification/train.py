from diffusion import GaussianDiffusion, Trainer, get_dataset
import torchvision
import os
import errno
import shutil
import argparse
from diffusion.model.get_model import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--train_steps', default=700000, type=int)
parser.add_argument('--save_folder', default='/cmlscratch/hmchu/cold_diff/', type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--dataset_folder', default='./root_cifar10', type=str)
parser.add_argument('--random_aug', action='store_true')
parser.add_argument('--output_mean_scale', action='store_true')
parser.add_argument('--exp_name', default='', type=str)
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--model', default='UnetConvNext', type=str)

parser.add_argument('--forward_process_type', default='Snow')

# Decolor args
parser.add_argument('--decolor_routine', default='Constant')
parser.add_argument('--decolor_ema_factor', default=0.9, type=float)
parser.add_argument('--decolor_total_remove', action='store_true')
parser.add_argument('--to_lab', action='store_true')

parser.add_argument('--loss_type', type=str, default='l1')
parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--load_model_steps', default=-1, type=int)

# Snow arg
parser.add_argument('--snow_level', default=1, type=int)
parser.add_argument('--random_snow', action='store_true')
parser.add_argument('--single_snow', action='store_true')
parser.add_argument('--fix_brightness', action='store_true')

parser.add_argument('--resolution', default=-1, type=int)

args = parser.parse_args()

assert len(args.exp_name) > 0
args.save_folder = os.path.join(args.save_folder, args.exp_name)
print(args)

if args.resume_training:
    if args.load_model_steps == -1:
        args.load_path = os.path.join(args.save_folder, 'model.pt')
    else:
        args.load_path = os.path.join(args.save_folder, f'model_{args.load_model_steps}.pt')
    print(f'resume from checkpoint stored at {args.load_path}')

with_time_emb = not args.remove_time_embed

model = get_model(args, with_time_emb=with_time_emb).cuda()
model_one_shot = get_model(args, with_time_emb=False).cuda()



image_size = get_dataset.get_image_size(args.dataset)
if args.resolution != -1:
    image_size = (args.resolution, args.resolution)

use_torchvison_dataset = False
if 'cifar10' in args.dataset:
    use_torchvison_dataset = True
    args.dataset = 'cifar10_train'

if image_size[0] <= 64:
    train_batch_size = 32
elif image_size[0] > 64:
    train_batch_size = 16


diffusion = GaussianDiffusion(
    model,
    image_size = image_size,
    device_of_kernel = 'cuda',
    channels = 3,
    one_shot_denoise_fn=model_one_shot,
    timesteps = args.time_steps,   # number of steps
    loss_type = args.loss_type,    # L1 or L2
    train_routine = args.train_routine,
    sampling_routine = args.sampling_routine,
    forward_process_type=args.forward_process_type,
    decolor_routine=args.decolor_routine,
    decolor_ema_factor=args.decolor_ema_factor,
    decolor_total_remove=args.decolor_total_remove,
    snow_level=args.snow_level,
    single_snow=args.single_snow,
    batch_size=train_batch_size,
    random_snow=args.random_snow,
    to_lab=args.to_lab,
    load_path=args.load_path,
    results_folder=args.save_folder,
    fix_brightness=args.fix_brightness,
).cuda()

trainer = Trainer(
    diffusion,
    args.dataset_folder,
    image_size = image_size,
    train_batch_size = train_batch_size,
    train_lr = 2e-5,
    train_num_steps = args.train_steps,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
    random_aug=args.random_aug,
    torchvision_dataset=use_torchvison_dataset,
    dataset = f'{args.dataset}',
    to_lab=args.to_lab,
)

trainer.train()
