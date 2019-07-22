# Copyright (c) 2019-present, Zhiqiang Wang.

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
        print(net_output.size())

    def compute_loss(self):
        pass

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
