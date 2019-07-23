# Copyright (c) 2019-present, Zhiqiang Wang.

from torchvision.models.detection.roi_heads import fastrcnn_loss
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('fasterrcnn_loss')
class FasterRCNNLoss(FairseqCriterion):

    def __init__(self, args, task):
        super(FairseqCriterion, self).__init__()
        self.args = args

    def forward(self, model, sample, reduction='mean'):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(sample['image'], sample['target'])
        rpn_proposals = model.get_rpn_proposals(net_output)
        box_detections = model.get_box_detections(net_output)

        # compute proposal losses
        anchors = model.rpn.get_anchors(rpn_proposals)
        objectness = model.rpn.get_objectness(rpn_proposals)
        pred_bbox_deltas = model.rpn.get_pred_bbox_deltas(rpn_proposals)
        labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, sample['target'])
        regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)

        loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets,
        )

        # compute detector loss
        labels = model.roi_heads.get_labels(box_detections)
        regression_targets = model.roi_heads.get_regression_targets(box_detections)
        class_logits = model.roi_heads.get_class_logits(box_detections)
        box_regression = model.roi_heads.get_box_regression(box_detections)

        loss_classifier, loss_box_reg = fastrcnn_loss(
            class_logits, box_regression, labels, regression_targets,
        )

        losses = {
            # proposal losses
            'loss_objectness': loss_objectness,
            'loss_rpn_box_reg': loss_rpn_box_reg,
            # detector loss
            'loss_classifier': loss_classifier,
            'loss_box_reg': loss_box_reg,
        }
        return losses

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        assert len(logging_outputs) == 1
        log = logging_outputs[0]
        loss = log.get('loss', 0)
        ntokens = log.get('ntokens', 0)
        batch_sizes = log.get('nsentences', 0)
        sample_size = log.get('sample_size', 0)
        agg_output = {
            'loss': loss,
            'ntokens': ntokens,
            'nsentences': batch_sizes,
            'sample_size': sample_size,
        }
        return agg_output
