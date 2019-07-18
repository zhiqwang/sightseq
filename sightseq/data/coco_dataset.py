# Copyright (c) 2019-present, Zhiqiang Wang.

import os
import numpy as np

from pycocotools.coco import COCO
from fairseq.data import FairseqDataset

from sightseq.data.data_utils import default_loader


class CocoDetectionDataset(FairseqDataset):
    def __init__(
        self, image_root, annotation_file,
        transforms=None, loader=default_loader,
    ):
        self.image_root = image_root
        self.coco = COCO(annotation_file)
        self.image_ids = list(sorted(self.coco.imgs.keys()))

        self.transforms = transforms
        self.loader = loader

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        image_id = self.image_ids[index]
        ann_ids = coco.getAnnIds(imgIds=image_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(image_id)[0]['file_name']

        image = self.loader(os.path.join(self.image_root, path))
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return {
            'id': image_id,
            'source': image,
            'target': target,
        }

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        pass

    def ordered_indices(self):
        """
        Return an ordered list of indices. Batches will be constructed based
        on this order.
        """
        if self.shuffle:
            return np.random.permutation(len(self))
        else:
            return np.arange(len(self))
