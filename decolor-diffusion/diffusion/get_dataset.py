import numpy as np
from torchvision import transforms, utils
from torchvision import datasets

def get_transform(image_size, random_aug=False, resize=False):
    if image_size[0] == 64:
        transform_list = [
            transforms.CenterCrop((128,128)),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ]
    elif not random_aug:
        transform_list = [
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ]
        if resize:
            transform_list = [transforms.Resize(image_size)] + transform_list
        T = transforms.Compose(transform_list)
    else:
        s = 1.0
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        T = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    return T

def get_image_size(name):
    if 'cifar10' in name:
        return (32, 32)
    if 'celebA' in name:
        return (128, 128)
    if 'flower' in name:
        return (128, 128)

def get_dataset(name, folder, image_size, random_aug=False):
    print(folder)
    if name == 'cifar10_train':
        return datasets.CIFAR10(folder, train=True, transform=get_transform(image_size, random_aug=random_aug))
    if name == 'cifar10_test':
        return datasets.CIFAR10(folder, train=False, transform=get_transform(image_size, random_aug=random_aug))
    if name == 'CelebA_train':
        return datasets.CelebA(folder, split='train', transform=get_transform(image_size, random_aug=random_aug), download=True)
    if name == 'CelebA_test':
        return datasets.CelebA(folder, split='test', transform=get_transform(image_size, random_aug=random_aug))
    if name == 'flower_train':
        return datasets.Flowers102(folder, split='train', transform=get_transform(image_size, random_aug=random_aug, resize=True), download=True)
    if name == 'flower_test':
        return datasets.Flowers102(folder, split='test', transform=get_transform(image_size, random_aug=random_aug, resize=True), download=True)

