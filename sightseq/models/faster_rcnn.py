# Copyright (c) 2019-present, Zhiqiang Wang.

from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)


@register_model('faster_rcnn')
class FasterRCNNModel(FairseqEncoderDecoderModel):
    """
    Args:
        encoder (TextRecognitionEncoder): the encoder
        decoder (LSTMDecoder): the decoder

    Attention model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.text_recognition_attn_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--backbone', default='resnet50',
                            help='CNN backbone architecture. (default: resnet50)')
        parser.add_argument('--pretrained', action='store_true', help='pretrained')

        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        pass

    def forward(self, src_tokens, prev_output_tokens, **kwargs):
        pass

    def extract_features(self, src_tokens, prev_output_tokens, **kwargs):
        pass


@register_model_architecture('faster_rcnn', 'faster_rcnn')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.backbone = getattr(args, 'backbone', 'resnet50')
    args.pretrained = getattr(args, 'pretrained', False)
