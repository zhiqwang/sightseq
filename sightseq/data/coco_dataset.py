# Copyright (c) 2019-present, Zhiqiang Wang

import os
import numpy as np
import torch
from pycocotools.coco import COCO
from fairseq.data import FairseqDataset

from sightseq.data.data_utils import default_loader


def collate(samples):
    """collate samples of images and targets."""
    if len(samples) == 0:
        return {}

    id = torch.LongTensor([s['id'] for s in samples])
    images = [s['image'] for s in samples]
    targets = [s['target'] for s in samples]
    ntokens = sum(len(t['labels']) for t in targets)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'image': images,
        'target': targets,
    }
    return batch


class CocoDetectionDataset(FairseqDataset):
    def __init__(
        self, image_root, annotation_file,
        shuffle=True, transforms=None, loader=default_loader,
    ):
        self.image_root = image_root
        print('| loading coco annotation file...')
        self.coco = COCO(annotation_file)
        self.image_ids = list(sorted(self.coco.imgs.keys()))

        self.shuffle = shuffle
        self.transforms = transforms
        self.loader = loader

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
        tgt_length = len(target['labels'])

        return {
            'id': image_id,
            'image': image,
            'target': target,
            'target_length': tgt_length,
        }

    def __len__(self):
        return len(self.image_ids)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        return collate(samples)

    def ordered_indices(self):
        """
        Return an ordered list of indices. Batches will be constructed based
        on this order.
        """
        if self.shuffle:
            return np.random.permutation(len(self))
        else:
            return np.arange(len(self))

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self[index]['target_length']

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self[index]['target_length']
