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
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
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
        b_channel1, g_channel1, r_channel1 = cv2.split(img1t)
        b_channel2, g_channel2, r_channel2 = cv2.split(img2t)
        b_channel3, g_channel3, r_channel3 = cv2.split(img3t)
        img1d= cv2.cvtColor(img1d,cv2.COLOR_GRAY2RGB)
        img2d= cv2.cvtColor(img2d,cv2.COLOR_GRAY2RGB)
        img3d= cv2.cvtColor(img3d,cv2.COLOR_GRAY2RGB)
        d_channel1, d_channel1, d_channel1 = cv2.split(img1d)
        d_channel2, d_channel2, d_channel2 = cv2.split(img2d)
        d_channel3, d_channel3, d_channel3 = cv2.split(img3d)

        depth1 = np.ones(d_channel2.shape, dtype=d_channel1.dtype)
        depth2 = np.ones(d_channel1.shape, dtype=d_channel2.dtype)
        depth3 = np.ones(d_channel3.shape, dtype=d_channel3.dtype)
        img1 = cv2.merge((b_channel1, g_channel1, r_channel1, depth1), cv2.COLOR_BGRA2RGBA)
        img2 = cv2.merge((b_channel2, g_channel2, r_channel2, depth2), cv2.COLOR_BGRA2RGBA)
        img3 = cv2.merge((b_channel3, g_channel3, r_channel3, depth3), cv2.COLOR_BGRA2RGBA)
        #img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2RGBA)
        #cv2.imwrite("1.png", img1)
        #cv2.waitKey(0)
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        img3 = Image.fromarray(img3)
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
