from pycave.bayes import GaussianMixture
import torchvision
import argparse
from Fid import calculate_fid_given_samples
import torch
import random

from resolution_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

seed_value=123457
torch.manual_seed(seed_value) # cpu  vars
random.seed(seed_value) # Python
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--sample_steps', default=None, type=int)
parser.add_argument('--save_folder', default='./results_cifar10', type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--data_path', default='./root_celebA_128_train_new/', type=str)
parser.add_argument('--resolution_routine', default='Incremental', type=str)
parser.add_argument('--test_type', default='train_data', type=str)
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--gmm_size', default=8, type=int)
parser.add_argument('--gmm_cluster', default=10, type=int)
parser.add_argument('--gmm_sample_at', default=1, type=int)
parser.add_argument('--bs', default=32, type=int)
parser.add_argument('--discrete', action="store_true")
parser.add_argument('--noise', default=0, type=float)

args = parser.parse_args()
print(args)


img_path=None
if 'train' in args.test_type:
    img_path = args.data_path
elif 'test' in args.test_type:
    img_path = args.data_path
print("Img Path is ", img_path)


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
    timesteps = args.time_steps,   # number of steps
    loss_type = 'l1',    # L1 or L2
    resolution_routine=args.resolution_routine,
    train_routine=args.train_routine,
    sampling_routine = args.sampling_routine
).cuda()

import torch
diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

trainer = Trainer(
    diffusion,
    img_path,
    image_size = 128,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                     # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
    shuffle=False,
    dataset = 'celebA'
)

if args.test_type == 'train_data':
    trainer.test_from_data('train', s_times=args.sample_steps)

elif args.test_type == 'test_fid_distance_decrease_from_manifold':
    trainer.fid_distance_decrease_from_manifold(calculate_fid_given_samples, start=0, end=None)

elif args.test_type == 'train_fid_distance_decrease_from_manifold':
    trainer.fid_distance_decrease_from_manifold(calculate_fid_given_samples, start=0, end=None)

elif args.test_type == 'train_distribution_mean_blur_torch_gmm_ablation':
    trainer.sample_as_a_mean_blur_torch_gmm_ablation(GaussianMixture, siz=args.gmm_size, ch=3, clusters=args.gmm_cluster, sample_at=args.gmm_sample_at, noise=args.noise)

