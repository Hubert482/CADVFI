import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import cv2

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
            #transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor()
        ])
        

    def __getitem__(self, index):
        if self.training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']

        # Load images
        img1t = cv2.imread(imgpaths[0])
        img2t = cv2.imread(imgpaths[1])
        img3t = cv2.imread(imgpaths[2])
        img1d = cv2.imread(imgpaths[0][:-4]+"_depth.png", cv2.IMREAD_GRAYSCALE)
        img2d = cv2.imread(imgpaths[1][:-4]+"_depth.png", cv2.IMREAD_GRAYSCALE)
        img3d = cv2.imread(imgpaths[2][:-4]+"_depth.png", cv2.IMREAD_GRAYSCALE)

        img1b, img1g, img1r = cv2.split(img1t)
        img2b, img2g, img2r = cv2.split(img2t)
        img3b, img3g, img3r = cv2.split(img3t)
        #imd = cv2.merge([depth , depth , depth ])

        img1 = cv2.merge([img1b, img1g, img1r, img1d], cv2.COLOR_BGRA2RGBA)
        img2 = cv2.merge((img2b, img2g, img2r, img2d), cv2.COLOR_BGRA2RGBA)
        img3 = cv2.merge((img3b, img3g, img3r, img3d), cv2.COLOR_BGRA2RGBA)
        #img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2RGBA)
        #cv2.imwrite("1.png", img1)
        #us, us, us, depth = cv2.split(img1)
        #imd = cv2.merge([depth , depth , depth ])
        #cv2.imwrite(f"depth.png", img1)
        #cv2.waitKey(0)
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        img3 = Image.fromarray(img3)
        #image_without_alpha = img[:,:,:3]
        #img = cv2.cvtColor(image_without_alpha, cv2.COLOR_RGBA2BGRA)

        # Data augmentation
        if self.training:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            img1 = self.transforms(img1)
            random.seed(seed)
            img2 = self.transforms(img2)
            random.seed(seed)
            img3 = self.transforms(img3)
            # Random Temporal Flip
            if random.random() >= 0.5:
                img1, img3 = img3, img1
                imgpaths[0], imgpaths[2] = imgpaths[2], imgpaths[0]
        else:
            T = transforms.ToTensor()
            img1 = T(img1)
            img2 = T(img2)
            img3 = T(img3)

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
