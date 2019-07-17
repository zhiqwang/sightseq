# Copyright (c) 2019-present, Zhiqiang Wang.

import os

from fairseq.tasks import register_task

from sightseq.tasks.sightseq_task import SightseqTask
from sightseq.data import CocoDetectionDataset, CocoDictionary
from sightseq.data.coco_utils import ConvertCocoPolysToMask
import sightseq.data.transforms as T


@register_task('object_detection')
class ObjectDetectionTask(SightseqTask):
    """
    Train a object detection model.

    Args:
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target text
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        # fmt: on

    def __init__(self, args, tgt_dict, transforms=None):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.transforms = transforms

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return CocoDictionary.load(filename)

    @classmethod
    def build_transforms(cls, args):
        transforms = []
        transforms.append(T.ToTensor())

        return T.Compose(transforms)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        tgt_dict = cls.load_dictionary(os.path.join(args.data, 'dict.txt'))
        print('| target dictionary: {} types'.format(len(tgt_dict)))

        # build transforms
        transforms = cls.build_transforms(args)

        return cls(args, tgt_dict, transforms=transforms)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # Read input images and targets
        img_folder = os.path.join(self.args.data, '{}2017'.format(split))
        ann_file = os.path.join(self.args.data, 'annotations', 'instances_{}2017.json'.format(split))

        t = [ConvertCocoPolysToMask()]

        if self.transforms is not None:
            t.append(self.transforms)
        if split == 'train':
            t.append(T.RandomHorizontalFlip(0.5))
        transforms = T.Compose(t)

        self.dataset[split] = CocoDetectionDataset(img_folder, ann_file, transforms=transforms)

        # if split == 'train':
        #     self.dataset[split].remove_images_without_annotations()

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
