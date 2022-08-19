from resolution_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, Model
import torchvision
import os
import errno
import shutil
import argparse
import torch
import random
from Fid import calculate_fid_given_samples

seed_value=123457
torch.manual_seed(seed_value) # cpu  vars
random.seed(seed_value) # Python
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
parser.add_argument('--save_folder', default='./results_cifar10', type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--data_path', default='./root_cifar10_test/', type=str)
parser.add_argument('--test_type', default='train_data', type=str)
parser.add_argument('--resolution_routine', default='Incremental', type=str)
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")


args = parser.parse_args()
print(args)

img_path=None
if args.test_type == 'train_distribution_cov_vector':
    img_path = args.data_path
elif 'train' in args.test_type:
    img_path = args.data_path
elif 'test' in args.test_type:
    img_path = args.data_path

print("Img Path is ", img_path)


model = Model(resolution=32,
              in_channels=3,
              out_ch=3,
              ch=128,
              ch_mult=(1,2,2,2),
              num_res_blocks=2,
              attn_resolutions=(16,),
              dropout=0.01).cuda()


diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    device_of_kernel = 'cuda',
    channels = 3,
    timesteps = args.time_steps,   # number of steps
    loss_type = 'l1',    # L1 or L2
    resolution_routine=args.resolution_routine,
    train_routine=args.train_routine,
    sampling_routine = args.sampling_routine
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
    load_path = args.load_path,
    shuffle=True
)

if args.test_type == 'train_data':
    trainer.test_from_data('train', s_times=args.sample_steps)

elif args.test_type == 'train_data_dropout':
    trainer.test_from_data_dropout('train_drop', s_times=args.sample_steps)

elif args.test_type == 'test_data_dropout':
    trainer.test_from_data_dropout('test_drop', s_times=args.sample_steps)

elif args.test_type == 'test_data':
    trainer.test_from_data('test', s_times=args.sample_steps)

elif args.test_type == 'mixup_train_data':
    trainer.test_with_mixup('train')

elif args.test_type == 'mixup_test_data':
    trainer.test_with_mixup('test')

elif args.test_type == 'test_random':
    trainer.test_from_random('test_random')

elif args.test_type == 'train_random':
    trainer.test_from_random('train_random')

elif args.test_type == 'train_distribution_cov_vector':
    trainer.sample_as_a_vector_cov(start=0, end=None, siz=4, ch=3)

elif args.test_type == 'test_fid_distance_decrease_from_manifold':
    trainer.fid_distance_decrease_from_manifold(calculate_fid_given_samples, start=0, end=None)

elif args.test_type == 'train_fid_distance_decrease_from_manifold':
    trainer.fid_distance_decrease_from_manifold(calculate_fid_given_samples, start=0, end=None)

elif args.test_type == 'test_paper_invert_section_images':
    trainer.paper_invert_section_images()

elif args.test_type == 'test_paper_showing_diffusion_images':
	trainer.paper_showing_diffusion_images()

elif args.test_type == 'test_paper_showing_diffusion_imgs_og':
    trainer.paper_showing_diffusion_imgs_og()
