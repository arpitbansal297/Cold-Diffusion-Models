import torchvision
import os
import errno
import shutil
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


CelebA_folder = "Path" # change this to folder which has CelebA data

############################################# MNIST ###############################################
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


trainset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True)
root = './root_mnist_test/'
del_folder(root)
create_folder(root)

for i in range(10):
    lable_root = root + str(i) + '/'
    create_folder(lable_root)

for idx in range(len(trainset)):
    img, label = trainset[idx]
    print(idx)
    img.save(root + str(label) + '/' + str(idx) + '.png')


############################################# Cifar10 ###############################################
trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True)
root = './root_cifar10/'
del_folder(root)
create_folder(root)

for i in range(10):
    lable_root = root + str(i) + '/'
    create_folder(lable_root)

for idx in range(len(trainset)):
    img, label = trainset[idx]
    print(idx)
    img.save(root + str(label) + '/' + str(idx) + '.png')


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


############################################# CelebA ###############################################
root_train = './root_celebA_128_train_new/'
root_test = './root_celebA_128_test_new/'
del_folder(root_train)
create_folder(root_train)

del_folder(root_test)
create_folder(root_test)

exts = ['jpg', 'jpeg', 'png']
folder = CelebA_folder
paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

for idx in range(len(paths)):
    img = Image.open(paths[idx])
    print(idx)
    if idx < 0.9*len(paths):
        img.save(root_train + str(idx) + '.png')
    else:
        img.save(root_test + str(idx) + '.png')