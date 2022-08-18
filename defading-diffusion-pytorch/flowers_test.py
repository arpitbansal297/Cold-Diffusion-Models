from defading_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from Fid import calculate_fid_given_samples
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
        if idx%10 == 0:
            img.save(root_test + str(idx) + '.png')
        else:
            img.save(root_train + str(idx) + '.png')


parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--sample_steps', default=None, type=int)
parser.add_argument('--blur_std', default=0.1, type=float)
parser.add_argument('--save_folder', default='./flowers_paper_defade', type=str)
parser.add_argument('--load_path', default="/cmlscratch/eborgnia/cold_diffusion/paper_defading_random_flowers_1/model.pt", type=str)
parser.add_argument('--test_type', default='test_paper_invert_section_images', type=str)
parser.add_argument('--fade_routine', default='Random_Incremental', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--discrete', action="store_true")
parser.add_argument('--residual', action="store_true")
args = parser.parse_args()
print(args)

img_path=None
if 'train' in args.test_type:
    img_path = './root_flower_train/'
elif 'test' in args.test_type:
    img_path = './root_flower_test/'

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
    timesteps=args.time_steps,
    loss_type='l1',
    kernel_std=args.blur_std,
    fade_routine=args.fade_routine,
    sampling_routine=args.sampling_routine,
    discrete=args.discrete
).cuda()

trainer = Trainer(
    diffusion,
    img_path,
    image_size=64,
    train_batch_size=32,
    train_lr=2e-5,
    train_num_steps=700000,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    fp16=False,
    results_folder=args.save_folder,
    load_path=args.load_path,
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
