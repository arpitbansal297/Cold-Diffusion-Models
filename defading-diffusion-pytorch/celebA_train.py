from defading_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch
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
        if exc.errno != errno.EEXIST:
            raise
        pass


parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=100, type=int)
parser.add_argument('--train_steps', default=700000, type=int)
parser.add_argument('--save_folder', default='test_speed_celebA', type=str)
parser.add_argument('--kernel_std', default=0.1, type=float)
parser.add_argument('--initial_mask', default=11, type=int)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--data_path', default='/cmlscratch/bansal01/spring_2022/Cold-Diffusion/deblurring-diffusion-pytorch/root_celebA_128_train_new/', type=str)
parser.add_argument('--fade_routine', default="Incremental", type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--discrete', action="store_true")
parser.add_argument('--image_size', default=64, type=int)
parser.add_argument('--dataset', default=None, type=str)
args = parser.parse_args()
print(args)

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=3,
    with_time_emb=not args.remove_time_embed,
    residual=args.residual
).cuda()

# model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

diffusion = GaussianDiffusion(
    model,
    image_size=args.image_size,
    device_of_kernel='cuda',
    channels=3,
    timesteps=args.time_steps,
    loss_type='l1',
    fade_routine=args.fade_routine,
    kernel_std=args.kernel_std,
    initial_mask=args.initial_mask,
    sampling_routine=args.sampling_routine,
    discrete=args.discrete
).cuda()

diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

trainer = Trainer(
    diffusion,
    # '/cmlscratch/bansal01/CelebA_train_new/train/',
    # './root_celebA_128_train/',
    args.data_path,
    image_size=args.image_size,
    train_batch_size=100,
    train_lr=2e-5,
    train_num_steps=args.train_steps,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    fp16=False,
    results_folder=args.save_folder,
    load_path=args.load_path,
    dataset=args.dataset
)
trainer.train()
