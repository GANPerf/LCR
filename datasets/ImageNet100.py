import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import torch
import torchvision
#https://blog.csdn.net/Andrew_SJ/article/details/111335319
class ImageNet100():
    def __init__(self, data_dir, train=True, transform=None):
        self.root = os.path.expanduser(data_dir)
        self.transform = transform
        #self.loader = default_loader
        self.train = train
        # self.classes=None
        # self.targets = None
        l=[]
        l.append(torchvision.datasets.ImageFolder(os.path.join(data_dir,'train.X1'),transform=transform))
        l.append(torchvision.datasets.ImageFolder(os.path.join(data_dir,'train.X2'), transform=transform))
        l.append(torchvision.datasets.ImageFolder(os.path.join(data_dir,'train.X3'), transform=transform))
        l.append(torchvision.datasets.ImageFolder(os.path.join(data_dir,'train.X4'), transform=transform))
        train_datasets=torch.utils.data.ConcatDataset(l)
        train_size = int(0.8 * len(train_datasets))
        test_size = len(train_datasets) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(train_datasets, [train_size, test_size])
        self.train_dataset = train_dataset.dataset
        self.test_dataset=test_dataset.dataset

    #base_folder = 'CUB_200_2011/images'
    #url = 'https://data.caltech.edu/tindfiles/serve/1239ea37-e132-42ee-8c09-c383bb54e7ff/' #''http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    def getdata(self):
        if self.train:
            return self.train_dataset
        else: return self.test_dataset

    #filename = 'CUB_200_2011.tgz'
    #tgz_md5 = '97eceeb196236b17998738112f37df78'


class train_dataset_transformed(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

 #train_dataset = train_dataset_transformed(train_dataset, transform=train_transform)
