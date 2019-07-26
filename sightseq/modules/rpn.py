# Copyright (c) 2019-present, Zhiqiang Wang.

from torchvision.models.detection.rpn import (
    RegionProposalNetwork,
    concat_box_prediction_layers,
)


class RPN(RegionProposalNetwork):
    """
    Implements Region Proposal Network (RPN).

    Arguments:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

    """

    def __init__(
        self, anchor_generator, head,
        fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction,
        pre_nms_top_n, post_nms_top_n, nms_thresh,
    ):
        super().__init__(
            anchor_generator, head,
            fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction,
            pre_nms_top_n, post_nms_top_n, nms_thresh,
        )

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (List[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level = [o[0].numel() for o in objectness]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(
            objectness, pred_bbox_deltas,
        )
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        return {
            'anchors': anchors,
            'objectness': objectness,
            'pred_bbox_deltas': pred_bbox_deltas,
            'boxes': boxes,
            'scores': scores,
        }

    @staticmethod
    def get_anchors(net_output):
        anchors = net_output['anchors']
        return anchors

    @staticmethod
    def get_objectness(net_output):
        objectness = net_output['objectness']
        return objectness

    @staticmethod
    def get_pred_bbox_deltas(net_output):
        pred_bbox_deltas = net_output['pred_bbox_deltas']
        return pred_bbox_deltas

    @staticmethod
    def get_boxes(net_output):
        boxes = net_output['boxes']
        return boxes

    @staticmethod
    def get_scores(net_output):
        scores = net_output['scores']
        return scores
