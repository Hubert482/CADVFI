import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class VimeoTriplet(Dataset):
    def __init__(self, data_root, is_training):
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, 'sequences')
        self.training = is_training

        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()
        
        self.transforms = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor()
        ])
        

    def __getitem__(self, index):
        if self.training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png', imgpath + '/im4.png', imgpath + '/im5.png', imgpath + '/im6.png', imgpath + '/im7.png']
        
        
        # Load images
        img1 = Image.open(imgpaths[0])
        img2 = Image.open(imgpaths[1])
        img3 = Image.open(imgpaths[2])
        img4 = Image.open(imgpaths[3])
        img5 = Image.open(imgpaths[4])
        img6 = Image.open(imgpaths[5])
        #img7 = Image.open(imgpaths[6])

        # Data augmentation
        if self.training:
            if random.random() >= 0.5:
                reverse_data=True
                seed = random.randint(0, 2**32)
                random.seed(seed)
                img1 = self.transforms(img1)
                random.seed(seed)
                img2 = self.transforms(img2)
                random.seed(seed)
                img3 = self.transforms(img3)
                #print(reverse_data)
            else:
                reverse_data=False
                seed = random.randint(0, 2**32)
                random.seed(seed)
                img1 = self.transforms(img4)
                random.seed(seed)
                img2 = self.transforms(img5)
                random.seed(seed)
                img3 = self.transforms(img6)
                #print(reverse_data)


            # Random Temporal Flip
            #if random.random() >= 0.5:
            #    img1, img3 = img3, img1
            #    imgpaths[0], imgpaths[2] = imgpaths[2], imgpaths[0]
        else:
            T = transforms.ToTensor()
            img1 = T(img1)
            #random.seed(seed)
            img2 = T(img2)
            #random.seed(seed)
            img3 = T(img3)
            #img4 = T(img4)
            #img5 = T(img5)
            #img6 = T(img6)
            #img7 = T(img7)
             #, img4, img5, img6, img7
             
        imgs = [img1, img2, img3]
        
        return imgs, imgpaths

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)
        return 0


def get_loader(mode, data_root, batch_size, shuffle, num_workers, test_mode=None):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = VimeoTriplet(data_root, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
