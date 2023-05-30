import numpy as np
#  Reading data
import scipy.misc
from matplotlib.pyplot import imread
import cv2
import os
from PIL import Image
from torchvision import transforms
import torch

class CUB():
    def __init__(self, root, is_train=True, data_len=None,transform=None, target_transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.target_transform = target_transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        train_class_file = open(os.path.join(self.root, 'classes.txt'))
        #  Picture index
        img_name_list = []
        for line in img_txt_file:
            #  The last character is a newline character
            img_name_list.append(line[:-1].split(' ')[-1])

        #  Tag Index , Each corresponding label minus １, Tag value from 0 Start
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_class_list=[]
        for line in train_class_file:
            train_class_list.append(line[:-1].split('.')[-1] )

        #  Set up training and test sets
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))

        # zip Compress merge , Associate data with labels ( Training set or test set ) Corresponding compression
        # zip()  Function to take iteratable objects as parameters , Package the corresponding elements in the object into tuples ,
        #  And then return the objects made up of these tuples , The advantage is that it saves a lot of memory .
        #  We can use  list()  Convert to output list

        #  If  i  by  1, Then set it as the training set
        # １ As the training set ,０ For test set
        # zip Compress merge , Associate data with labels ( Training set or test set ) Corresponding compression
        #  If  i  by  1, Then set it as the training set
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        train_label_list = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        test_label_list = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
        if self.is_train:
            # scipy.misc.imread  The picture reads out as array type , namely numpy type
            self.train_img = [self.read_pic(os.path.join(self.root, 'images', train_file)) for train_file in
                              train_file_list[:data_len]]
            #  Read the training set label
            self.train_label = train_label_list
            self.targets=train_label_list
            self.classes=train_class_list
        if not self.is_train:
            self.test_img = [self.read_pic(os.path.join(self.root, 'images', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = test_label_list
            self.targets = test_label_list
            self.classes = train_class_list

    #  Data to enhance
    def __getitem__(self,index):
        #  Training set
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
        #  Test set
        else:
            img, target = self.test_img[index], self.test_label[index]

        if len(img.shape) == 2:
            #  Gray images are converted to three channels
            img = np.stack([img]*3,2)
        #  To  RGB  type
        img = Image.fromarray(img,mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)

    def read_pic(self, train_file ):
        try:
           img= cv2.imread(train_file)
           return img
        except Exception as e:
            print(train_file)
            print(e)

if __name__ == '__main__':
    ''' dataset = CUB(root='./CUB_200_2011') for data in dataset: print(data[0].size(),data[1]) '''
    #  With pytorch in DataLoader Read the data set in a way
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    dataset = CUB(root='./CUB_200_2011', is_train=False, transform=transform_train,)
    print(len(dataset))
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,
                                              drop_last=True)
    print(len(trainloader))