from defading_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch
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


create = 0

if create:
    root_train = './root_flower_train/'
    root_test = './root_flower_test/'
    del_folder(root_train)
    create_folder(root_train)
    del_folder(root_test)
    create_folder(root_test)
    exts = ['jpg', 'jpeg', 'png']
    folder = '/fs/cml-datasets/vgg_flowers/jpg/'
    paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
    for idx in range(len(paths)):
        img = Image.open(paths[idx])
        print(idx)
        if idx % 10 == 0:
            img.save(root_test + str(idx) + '.png')
        else:
            img.save(root_train + str(idx) + '.png')


parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--train_steps', default=700000, type=int)
parser.add_argument('--save_folder', default='./flowers_test_24', type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--fade_routine', default='Random_Incremental', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--discrete', action="store_true")
args = parser.parse_args()
print(args)


model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=3,
    with_time_emb=not args.remove_time_embed,
    residual=args.residual
).cuda()

model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

diffusion = GaussianDiffusion(
    model,
    image_size=64,
    device_of_kernel='cuda',
    channels=3,
    timesteps=args.time_steps,   # number of steps
    loss_type='l1',    # L1 or L2
    fade_routine=args.fade_routine,
    sampling_routine=args.sampling_routine,
    discrete=args.discrete
).cuda()

trainer = Trainer(
    diffusion,
    './root_flower_train/',
    image_size=64,
    train_batch_size=8,
    train_lr=2e-5,
    train_num_steps=args.train_steps,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    fp16=False,                       # turn on mixed precision training with apex
    results_folder=args.save_folder,
    load_path=args.load_path,
)

trainer.train()
