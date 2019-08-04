# Copyright (c) 2019-present, Zhiqiang Wang

import os
import torch

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
        parser.add_argument('data', metavar='FOLDER',
                            help='path to data directory')
        parser.add_argument('--pretrained', action='store_true', help='pretrained')
        parser.add_argument('--num-classes', type=int, default=-1, metavar='N',
                            help='number of output classes of the model (including the background).'
                                 ' If box_predictor is specified, num_classes should be None')
        parser.add_argument('--max-positions', default=2048, type=int,
                            help='max input length')
        parser.add_argument('--aspect-ratio-group-factor', type=int, default=0, metavar='N',
                            help='data sampler with aspect ratio group')
        # fmt: on

    def __init__(
        self, args, num_classes, transforms=None,
        rpn_anchor_generator=None, rpn_head=None,
        box_roi_pool=None, box_predictor=None, box_head=None,
        pretrained=False,
    ):
        super().__init__(args)
        self.num_classes = num_classes
        self.transforms = transforms
        self.rpn_anchor_generator = rpn_anchor_generator
        self.rpn_head = rpn_head
        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor
        self.box_head = box_head
        self.pretrained = pretrained

    @classmethod
    def build_transforms(cls, args):
        transforms = [ConvertCocoPolysToMask()]
        transforms.append(T.ToTensor())

        return transforms

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        # build transforms
        transforms = cls.build_transforms(args)
        return cls(
            args,
            num_classes=args.num_classes,
            transforms=transforms,
            pretrained=args.pretrained,
        )

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # Read input images and targets
        image_root = os.path.join(self.args.data, '{}2017'.format(split))
        annotation_file = os.path.join(self.args.data, 'annotations', 'instances_{}2017.json'.format(split))

        t = self.transforms or []
        if split == 'train':
            t.append(T.RandomHorizontalFlip(0.5))
        transforms = T.Compose(t)

        self.datasets[split] = CocoDetectionDataset(image_root, annotation_file, transforms=transforms)

        # if split == 'train':
        #     self.dataset[split].remove_images_without_annotations()

    def build_model(self, args):
        """
        Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
        task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        model = super().build_model(args)
        return model

    def build_generator(self, args):
        from sightseq.coco_generator import ObjectDetectionGenerator
        return ObjectDetectionGenerator()

    def valid_step(self, sample, model, criterion):
        model.train()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions*
        return self.args.max_positions
