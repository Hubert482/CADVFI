import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Video(Dataset):
    def __init__(self, data_root, fmt='png'):
        images = sorted(glob.glob(os.path.join(data_root, '*.%s' % fmt)))
        for im in images:
            try:
                float_ind = float(im.split('_')[-1][:-4])
            except ValueError:
                os.rename(im, '%s_%.06f.%s' % (im[:-4], 0.0, fmt))
        # re
        images = sorted(glob.glob(os.path.join(data_root, '*.%s' % fmt)))
        self.imglist = [[images[i], images[i+1]] for i in range(len(images)-1)]
        print('[%d] images ready to be loaded' % len(self.imglist))


    def __getitem__(self, index):
        imgpaths = self.imglist[index]
        img1t = cv2.imread(imgpaths[0])
        img2t = cv2.imread(imgpaths[1])
        #img3t = cv2.imread(imgpaths[2])
        img1d = cv2.imread(imgpaths[0][:-4]+"_depth.png", cv2.IMREAD_GRAYSCALE)
        img2d = cv2.imread(imgpaths[1][:-4]+"_depth.png", cv2.IMREAD_GRAYSCALE)
        #img3d = cv2.imread(imgpaths[2][:-4]+"_depth.png", cv2.IMREAD_GRAYSCALE)
        b_channel1, g_channel1, r_channel1 = cv2.split(img1t)
        b_channel2, g_channel2, r_channel2 = cv2.split(img2t)
        #b_channel3, g_channel3, r_channel3 = cv2.split(img3t)
        img1d= cv2.cvtColor(img1d,cv2.COLOR_GRAY2RGB)
        img2d= cv2.cvtColor(img2d,cv2.COLOR_GRAY2RGB)
        #img3d= cv2.cvtColor(img3d,cv2.COLOR_GRAY2RGB)
        d_channel1, d_channel1, d_channel1 = cv2.split(img1d)
        d_channel2, d_channel2, d_channel2 = cv2.split(img2d)
        #d_channel3, d_channel3, d_channel3 = cv2.split(img3d)

        depth1 = np.ones(d_channel2.shape, dtype=d_channel1.dtype)
        depth2 = np.ones(d_channel1.shape, dtype=d_channel2.dtype)
        #depth3 = np.ones(d_channel3.shape, dtype=d_channel3.dtype)
        img1 = cv2.merge((b_channel1, g_channel1, r_channel1, depth1), cv2.COLOR_BGRA2RGBA)
        img2 = cv2.merge((b_channel2, g_channel2, r_channel2, depth2), cv2.COLOR_BGRA2RGBA)
        #img3 = cv2.merge((b_channel3, g_channel3, r_channel3, depth3), cv2.COLOR_BGRA2RGBA)
        #img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2RGBA)
        #cv2.imwrite("1.png", img1)
        #cv2.waitKey(0)
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        #img3 = Image.fromarray(img3)
        # Load images


        T = transforms.ToTensor()
        img1 = T(img1)
        img2 = T(img2)

        imgs = [img1, img2] 
        meta = {'imgpath': imgpaths}
        return imgs, meta

    def __len__(self):
        return len(self.imglist)


def get_loader(mode, data_root, batch_size, img_fmt='png', shuffle=False, num_workers=0, n_frames=1):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = Video(data_root, fmt=img_fmt)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
