import os.path

import torch
import torchvision
from .random_dataset import RandomDataset
from .CUB2011 import Cub2011 as CUB
from .ImageNet100 import ImageNet100
#from .StandfordCars import StandfordCars


def get_dataset(dataset, data_dir, transform, train=True, download=False, debug_subset_size=None):
    if dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'stl10':
        dataset = torchvision.datasets.STL10(data_dir, split='train+unlabeled' if train else 'test', transform=transform, download=download)
    elif dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(data_dir, split='train' if train == True else 'val', transform=transform, download=download)
    elif dataset == 'imagenet100':
        data_builder = ImageNet100(data_dir, train = train,transform=transform)# torchvision.datasets.ImageFolder(data_dir , transform=transform) #+ 'train' if train == True else 'val',
        dataset=data_builder.getdata()
    elif dataset =='cub200':
        # dataset =     elif dataset == 'imagenet':
        # dataset= torchvision.datasets.ImageFolder(data_dir+ '/train' if train == True else data_dir+ '/test', transform=transform)
        # print('from datasets.imagefolders')
        #############dataset = CUB(data_dir, train = train,transform=transform,download=download)
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train') if train == True else os.path.join(data_dir, 'test'), transform=transform)
        # print('')
        # trainloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,
        #                                           drop_last=True)
    elif dataset=='stanfordcars':
        dataset= torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train') if train == True else os.path.join(data_dir, 'test'), transform=transform)#StandfordCars(data_dir,train=train,transform=transform)
    elif dataset == 'aircrafts':
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train') if train == True else os.path.join(data_dir, 'test'),
            transform=transform)
    elif dataset == 'random':
        dataset = RandomDataset()
    else:
        raise NotImplementedError
    #debug_subset_size=50
    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch
        dataset.classes = dataset.dataset.classes
        dataset.targets = dataset.dataset.targets
    return dataset