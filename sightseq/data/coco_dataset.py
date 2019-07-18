# Copyright (c) 2019-present, Zhiqiang Wang.

import torchvision


class CocoDetectionDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file, transforms=transforms)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return {
            'id': image_id,
            'image': img,
            'target': target,
        }
