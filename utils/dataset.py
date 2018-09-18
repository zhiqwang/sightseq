import os
import numpy as np

import torch
import skimage
import skimage.io as io
import cv2
import torch.utils.data as data
import glob


class digitsDataset(data.Dataset):
    """ Digits dataset."""
    def __init__(self, root_dir, transform=None):
        self.img_files = glob.glob('{}/*.jpg'.format(root_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img = io.imread(img_name)
        img = skimage.img_as_float32(img)

        if self.transform is not None:
            img = self.transform(img)

        label = img_name.split('_')[-1].split('.')[0]

        return img, label


class Normalize(object):
    """Normalize"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img -= self.mean
        img /= self.std
        return img


class Resize(object):
    """Resize."""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = cv2.resize(img, self.size)
        return img


class ToTensor(object):
    """change image to sequence."""
    def __call__(self, img):
        img = torch.from_numpy(img)
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.permute(2, 0, 1)
        return img


class ToGray(object):
    """ToGray"""
    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


class ToTensorRGBFlatten(object):
    """change image to sequence."""
    def __call__(self, img):
        seq_len = img.shape[1]
        img = img.transpose(1, 0, 2).reshape((seq_len, -1))
        img = torch.from_numpy(img)
        return img


class ToTensorGray(object):
    """change image to sequence."""
    def __call__(self, img):
        img = torch.from_numpy(img)
        img = img.permute(1, 0).contiguous()
        return img
