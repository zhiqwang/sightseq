# Copyright (c) 2019-present, Zhiqiang Wang.

import torch
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss


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

        result, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                    )
                )

        return result, losses
