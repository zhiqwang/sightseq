# Copyright (c) 2019-present, Zhiqiang Wang.

from collections import OrderedDict

import torch

from torchvision.ops import MultiScaleRoIAlign

from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RPNHead,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)

from sightseq.modules import RPN, RegionOfInterestHeads


@register_model('faster_rcnn')
class FasterRCNN(BaseFairseqModel):
    """
    Faster RCNN from `"Faster RCNN: Towards Real-Time Object Detection
    with Region Proposal Networks" (Ren, et al, 2015)
    <https://arxiv.org/abs/1506.01497>`_.

    Adopted from the torchvision implementation.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the
            RPN and computes detections / masks from it
        transform (nn.Module): performs the data transformation from
            the inputs to feed into the model

    Faster RCNN provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.faster_rcnn_parser
        :prog:
    """
    @classmethod
    def hub_models(cls):
        return {
            'fasterrcnn.resnet50.fpn.coco':
                'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
        }

    def __init__(self, backbone, rpn, roi_heads, transform):
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--backbone', default='resnet50',
                            help='CNN backbone architecture. (default: resnet50)')
        parser.add_argument('--pretrained', action='store_true', help='pretrained')
        # transform parameters
        parser.add_argument('--num-classes', type=int, metavar='N',
                            help='number of output classes of the'
                                 ' model (including the background). If box_predictor'
                                 ' is specified, num_classes should be None')
        parser.add_argument('--min-size', type=int, metavar='N',
                            help='minimum size of the image to be rescaled'
                                 ' before feeding it to the backbone')
        parser.add_argument('--max-size', type=int, metavar='N',
                            help='maximum size of the image to be rescaled'
                                 ' before feeding it to the backbone')
        parser.add_argument('--image-mean', type=tuple,
                            help='mean values used for input normalization')
        parser.add_argument('--image-std', type=tuple,
                            help='std values used for input normalization')
        # RPN parameters
        parser.add_argument('--rpn-pre-nms-top-n-train', type=int, metavar='N',
                            help='number of proposals to keep before'
                                 ' applying NMS during training')
        parser.add_argument('--rpn-pre-nms-top-n-test', type=int, metavar='N',
                            help='number of proposals to keep before'
                                 ' applying NMS during testing')
        parser.add_argument('--rpn-post-nms-top-n-train', type=int, metavar='N',
                            help='number of proposals to keep after'
                                 ' applying NMS during training')
        parser.add_argument('--rpn-post-nms-top-n-test', type=int, metavar='N',
                            help='number of proposals to keep after'
                                 ' applying NMS during testing')
        parser.add_argument('--rpn-nms-thresh', type=float, metavar='D',
                            help='NMS threshold used for postprocessing the RPN proposals')
        parser.add_argument('--rpn-fg-iou-thresh', type=float, metavar='D',
                            help='minimum IoU between the anchor and the GT box'
                                 ' so that they can be considered as positive'
                                 ' during training of the RPN')
        parser.add_argument('--rpn-bg-iou-thresh', type=float, metavar='D',
                            help='maximum IoU between the anchor and the GT box'
                                 ' so that they can be considered as negative'
                                 ' during training of the RPN')
        parser.add_argument('--rpn-batch-size-per-image', type=int, metavar='N',
                            help='number of anchors that are sampled during training'
                                 ' of the RPN for computing the loss')
        parser.add_argument('--rpn-positive-fraction', type=float, metavar='D',
                            help='proportion of positive anchors in a'
                                 ' mini-batch during training of the RPN')
        # Box parameters
        parser.add_argument('--box-score-thresh', type=float, metavar='D',
                            help='during inference, only return proposals with'
                                 ' a classification score greater than box_score_thresh')
        parser.add_argument('--box-nms-thresh', type=float, metavar='D',
                            help='NMS threshold for the prediction head. Used during inference')
        parser.add_argument('--box-detections-per-img', type=int, metavar='N',
                            help='maximum number of detections per image, for all classes')
        parser.add_argument('--box-fg-iou-thresh', type=float, metavar='D',
                            help='minimum IoU between the proposals and the GT box'
                                 ' so that they can be considered as positive during'
                                 ' training of the classification head')
        parser.add_argument('--box-bg-iou-thresh', type=float, metavar='D',
                            help='maximum IoU between the proposals and the GT box'
                                 ' so that they can be considered as negative during'
                                 ' training of the classification head')
        parser.add_argument('--box-batch-size-per-image', type=int, metavar='N',
                            help='number of proposals that are sampled during'
                                 ' training of the classification head')
        parser.add_argument('--box-positive-fraction', type=float, metavar='D',
                            help='proportion of positive proposals in a mini-batch'
                                 ' during training of the classification head')
        parser.add_argument('--bbox-reg-weights', type=tuple,
                            help='weights for the encoding/decoding of the bounding boxes')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        rpn_anchor_generator = task.rpn_anchor_generator
        box_roi_pool = task.box_roi_pool
        box_predictor = task.box_predictor
        rpn_head = task.rpn_head
        box_head = task.box_head

        # setup backbone
        pretrained_backbone = True
        if args.pretrained:
            # no need to download the backbone if pretrained is set
            pretrained_backbone = False
        backbone = resnet_fpn_backbone(args.backbone, pretrained_backbone)

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if args.num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels,
                rpn_anchor_generator.num_anchors_per_location()[0],
            )

        rpn_pre_nms_top_n = dict(training=args.rpn_pre_nms_top_n_train, testing=args.rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=args.rpn_post_nms_top_n_train, testing=args.rpn_post_nms_top_n_test)

        rpn = RPN(
            rpn_anchor_generator, rpn_head,
            args.rpn_fg_iou_thresh, args.rpn_bg_iou_thresh,
            args.rpn_batch_size_per_image, args.rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, args.rpn_nms_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=7,
                sampling_ratio=2,
            )

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size,
            )

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                args.num_classes,
            )

        roi_heads = RegionOfInterestHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            args.box_fg_iou_thresh, args.box_bg_iou_thresh,
            args.box_batch_size_per_image, args.box_positive_fraction,
            args.bbox_reg_weights, args.box_score_thresh,
            args.box_nms_thresh, args.box_detections_per_img,
        )

        if args.image_mean is None:
            args.image_mean = [0.485, 0.456, 0.406]
        if args.image_std is None:
            args.image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(
            args.min_size, args.max_size,
            args.image_mean, args.image_std,
        )

        return FasterRCNN(backbone, rpn, roi_heads, transform)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        rpn_proposals = self.rpn(images, features, targets)
        boxes = self.rpn.get_boxes(rpn_proposals)
        detections = self.roi_heads(features, boxes, images.image_sizes, targets)

        # post process when not training
        if not self.training:
            roi_heads_hypos = self.roi_heads.get_hypos(detections)
            detections = self.transform.postprocess(
                roi_heads_hypos, images.image_sizes, original_image_sizes,
            )

        return {
            'rpn_proposals': rpn_proposals,
            'box_detections': detections,
        }

    @staticmethod
    def get_rpn_proposals(net_output):
        rpn_proposals = net_output['rpn_proposals']
        return rpn_proposals

    @staticmethod
    def get_box_detections(net_output):
        box_detections = net_output['box_detections']
        return box_detections

    def max_positions(self):
        """Maximum length supported by the model."""
        return 1e6  # an arbitrary large number


@register_model_architecture('faster_rcnn', 'faster_rcnn')
def base_architecture(args):
    args.backbone = getattr(args, 'backbone', 'resnet50')
    args.pretrained = getattr(args, 'pretrained', False)
    args.num_classes = getattr(args, 'num_classes', None)
    args.min_size = getattr(args, 'min_size', 800)
    args.max_size = getattr(args, 'max_size', 1333)
    args.image_mean = getattr(args, 'image_mean', None)
    args.image_std = getattr(args, 'image_std', None)
    args.rpn_pre_nms_top_n_train = getattr(args, 'rpn_pre_nms_top_n_train', 2000)
    args.rpn_pre_nms_top_n_test = getattr(args, 'rpn_pre_nms_top_n_test', 1000)
    args.rpn_post_nms_top_n_train = getattr(args, 'rpn_post_nms_top_n_train', 2000)
    args.rpn_post_nms_top_n_test = getattr(args, 'rpn_post_nms_top_n_test', 1000)
    args.rpn_nms_thresh = getattr(args, 'rpn_nms_thresh', 0.7)
    args.rpn_fg_iou_thresh = getattr(args, 'rpn_fg_iou_thresh', 0.7)
    args.rpn_bg_iou_thresh = getattr(args, 'rpn_bg_iou_thresh', 0.3)
    args.rpn_batch_size_per_image = getattr(args, 'rpn_batch_size_per_image', 256)
    args.rpn_positive_fraction = getattr(args, 'rpn_positive_fraction', 0.5)
    args.box_score_thresh = getattr(args, 'box_score_thresh', 0.05)
    args.box_nms_thresh = getattr(args, 'box_nms_thresh', 0.5)
    args.box_detections_per_img = getattr(args, 'box_detections_per_img', 100)
    args.box_fg_iou_thresh = getattr(args, 'box_fg_iou_thresh', 0.5)
    args.box_bg_iou_thresh = getattr(args, 'box_bg_iou_thresh', 0.5)
    args.box_batch_size_per_image = getattr(args, 'box_batch_size_per_image', 512)
    args.box_positive_fraction = getattr(args, 'box_positive_fraction', 0.25)
    args.bbox_reg_weights = getattr(args, 'bbox_reg_weights', None)


@register_model_architecture('faster_rcnn', 'fasterrcnn_resnet50_fpn')
def fasterrcnn_resnet50_fpn(args):
    args.backbone = getattr(args, 'backbone', 'resnet50')
    base_architecture(args)
