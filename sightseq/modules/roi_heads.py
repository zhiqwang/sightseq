# Copyright (c) 2019-present, Zhiqiang Wang.

import torch
from torchvision.models.detection.roi_heads import RoIHeads


class RegionOfInterestHeads(RoIHeads):
    def __init__(
        self, box_roi_pool, box_head, box_predictor,
        # Faster R-CNN training
        fg_iou_thresh, bg_iou_thresh, batch_size_per_image,
        positive_fraction, bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh, nms_thresh, detections_per_img,
        # Mask
        mask_roi_pool=None, mask_head=None, mask_predictor=None,
        keypoint_roi_pool=None, keypoint_head=None, keypoint_predictor=None,
    ):
        super().__init__(
            box_roi_pool, box_head, box_predictor,
            fg_iou_thresh, bg_iou_thresh, batch_size_per_image,
            positive_fraction, bbox_reg_weights,
            score_thresh, nms_thresh, detections_per_img,
            mask_roi_pool, mask_head, mask_predictor,
            keypoint_roi_pool, keypoint_head, keypoint_predictor,
        )

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                assert t["boxes"].dtype.is_floating_point, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint:
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        if not self.training:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)

        return {
            'labels': labels,
            'regression_targets': regression_targets if self.training else None,
            'class_logits': class_logits,
            'box_regression': box_regression,
            'boxes': boxes if not self.training else None,
            'scores': scores if not self.training else None,
        }

    @staticmethod
    def get_labels(net_output):
        labels = net_output['labels']
        return labels

    @staticmethod
    def get_regression_targets(net_output):
        regression_targets = net_output['regression_targets']
        return regression_targets

    @staticmethod
    def get_class_logits(net_output):
        class_logits = net_output['class_logits']
        return class_logits

    @staticmethod
    def get_box_regression(net_output):
        box_regression = net_output['box_regression']
        return box_regression

    @staticmethod
    def get_boxes(net_output):
        boxes = net_output['boxes']
        return boxes

    @staticmethod
    def get_scores(net_output):
        scores = net_output['scores']
        return scores
