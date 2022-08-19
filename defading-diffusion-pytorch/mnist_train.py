from defading_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
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
    trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True)
    root = './root_mnist/'
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
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--train_steps', default=700000, type=int)
parser.add_argument('--save_folder', default='./_step_50_gaussian', type=str)
parser.add_argument('--blur_std', default=0.1, type=float)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--data_path', default='./root_mnist/', type=str)
parser.add_argument('--fade_routine', default='Random_Incremental', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str)
parser.add_argument('--discrete', action="store_true")
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")


args = parser.parse_args()
print(args)

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=1,
    with_time_emb=not args.remove_time_embed,
    residual=args.residual
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size=32,
    device_of_kernel='cuda',
    channels=1,
    timesteps=args.time_steps,
    loss_type='l1',
    fade_routine=args.fade_routine,
    kernel_std=args.blur_std,
    sampling_routine=args.sampling_routine,
    discrete=args.discrete
).cuda()

trainer = Trainer(
    diffusion,
    args.data_path,
    image_size=32,
    train_batch_size=32,
    train_lr=2e-5,
    train_num_steps=args.train_steps,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    fp16=False,
    results_folder=args.save_folder,
    load_path=args.load_path,
)

trainer.train()