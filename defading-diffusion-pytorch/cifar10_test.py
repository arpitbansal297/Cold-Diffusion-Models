from defading_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, Model
from Fid import calculate_fid_given_samples
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
            root='./data', train=False, download=True)
    root = './root_cifar10_test/'
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
parser.add_argument('--sample_steps', default=None, type=int)
parser.add_argument('--blur_std', default=0.1, type=float)
parser.add_argument('--save_folder', default='progression_cifar', type=str)
parser.add_argument('--load_path', default='/cmlscratch/eborgnia/cold_diffusion/paper_defading_random_1/model.pt', type=str)
parser.add_argument('--data_path', default='./root_cifar10_test/', type=str)
parser.add_argument('--test_type', default='test_paper_showing_diffusion_images_diff', type=str)
parser.add_argument('--fade_routine', default='Random_Incremental', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--discrete', action="store_true")
parser.add_argument('--residual', action="store_true")

args = parser.parse_args()
print(args)

img_path=None
if 'train' in args.test_type:
    img_path = args.data_path
elif 'test' in args.test_type:
    img_path = args.data_path

print("Img Path is ", img_path)

model = Model(resolution=32,
              in_channels=3,
              out_ch=3,
              ch=128,
              ch_mult=(1, 2, 2, 2),
              num_res_blocks=2,
              attn_resolutions=(16,),
              dropout=0.1).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    device_of_kernel = 'cuda',
    channels = 3,
    timesteps = args.time_steps,   # number of steps
    loss_type = 'l1',    # L1 or L2
    kernel_std=args.blur_std,
    fade_routine=args.fade_routine,
    sampling_routine = args.sampling_routine,
    discrete=args.discrete
).cuda()

trainer = Trainer(
    diffusion,
    img_path,
    image_size = 32,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path
)

if args.test_type == 'train_data':
    trainer.test_from_data('train', s_times=args.sample_steps)

elif args.test_type == 'test_data':
    trainer.test_from_data('test', s_times=args.sample_steps)

elif args.test_type == 'mixup_train_data':
    trainer.test_with_mixup('train')

elif args.test_type == 'mixup_test_data':
    trainer.test_with_mixup('test')

elif args.test_type == 'test_random':
    trainer.test_from_random('random')

elif args.test_type == 'test_fid_distance_decrease_from_manifold':
    trainer.fid_distance_decrease_from_manifold(calculate_fid_given_samples, start=0, end=None)

elif args.test_type == 'test_paper_invert_section_images':
    trainer.paper_invert_section_images()

elif args.test_type == 'test_paper_showing_diffusion_images_diff':
    trainer.paper_showing_diffusion_images()
