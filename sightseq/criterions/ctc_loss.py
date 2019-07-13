# Copyright (c) 2019-present, Zhiqiang Wang.

import torch
import torch.nn.functional as F

from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('ctc_loss')
class CTCLossCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super(FairseqCriterion, self).__init__()
        self.args = args
        self.blank_idx = task.target_dictionary.blank()

    def forward(self, model, sample, reduction='mean'):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss = self.compute_loss(model, net_output, sample, reduction=reduction)
        sample_size = sample['nsentences'] if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.item(),
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(
        self, model, net_output, sample,
        reduction='mean', zero_infinity=False,
    ):
        log_probs = model.get_normalized_probs(net_output, log_probs=True)
        targets = torch.cat(sample['target']).cpu()  # Expected targets to have CPU Backend
        target_lengths = sample['target_length']
        input_lengths = torch.full((sample['nsentences'],), log_probs.size(0), dtype=torch.int32)
        loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                          blank=self.blank_idx, reduction=reduction,
                          zero_infinity=zero_infinity)
        return loss

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
