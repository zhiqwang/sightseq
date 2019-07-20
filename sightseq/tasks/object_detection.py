# Copyright (c) 2019-present, Zhiqiang Wang.

import os

from fairseq.tasks import FairseqTask, register_task

from sightseq.data import CocoDetectionDataset
from sightseq.data.coco_utils import ConvertCocoPolysToMask
import sightseq.data.transforms as T


@register_task('object_detection')
class ObjectDetectionTask(FairseqTask):
    """
    Train a object detection model.

    Args:
        transforms:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        # fmt: on

    def __init__(
        self, args, transforms=None,
        rpn_anchor_generator=None, box_roi_pool=None,
        box_predictor=None, rpn_head=None, box_head=None,
    ):
        super().__init__(args)
        self.transforms = transforms
        self.rpn_anchor_generator = rpn_anchor_generator
        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor
        self.rpn_head = rpn_head
        self.box_head = box_head

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

        # build transforms
        transforms = cls.build_transforms(args)
        return cls(args, transforms=transforms)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # Read input images and targets
        image_root = os.path.join(self.args.data, '{}2017'.format(split))
        annotation_file = os.path.join(self.args.data, 'annotations', 'instances_{}2017.json'.format(split))

        t = [ConvertCocoPolysToMask()]

        if self.transforms is not None:
            t.append(self.transforms)
        if split == 'train':
            t.append(T.RandomHorizontalFlip(0.5))
        transforms = T.Compose(t)

        self.datasets[split] = CocoDetectionDataset(image_root, annotation_file, transforms=transforms)

        # if split == 'train':
        #     self.dataset[split].remove_images_without_annotations()
