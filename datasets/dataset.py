import os
import glob

import torch
import torch.utils.data as data
from datasets.datahelpers import default_loader


class DigitsDataset(data.Dataset):
    """ Digits dataset."""
    def __init__(self, img_path, transform=None, loader=default_loader):
        self.img_names = glob.glob('{}/*.jpg'.format(img_path))
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        image = self.loader(img_name)

        if self.transform is not None:
            image = self.transform(image)

        target = img_name.split('_')[-1].split('.')[0]

        return image, target
