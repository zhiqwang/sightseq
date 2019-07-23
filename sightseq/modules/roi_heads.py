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
        else:
            matched_idxs, labels, regression_targets = None, None, None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        return {
            'matched_idxs': matched_idxs,
            'labels': labels,
            'regression_targets': regression_targets,
            'class_logits': class_logits,
            'box_regression': box_regression,
        }

    def get_matched_idxs(self, net_output):
        matched_idxs = net_output['matched_idxs']
        return matched_idxs

    def get_labels(self, net_output):
        labels = net_output['labels']
        return labels

    def get_regression_targets(self, net_output):
        regression_targets = net_output['regression_targets']
        return regression_targets

    def get_class_logits(self, net_output):
        class_logits = net_output['class_logits']
        return class_logits

    def get_box_regression(self, net_output):
        box_regression = net_output['box_regression']
        return box_regression
