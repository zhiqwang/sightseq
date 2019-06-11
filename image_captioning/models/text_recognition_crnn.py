# Copyright (c) 2019-present, Zhiqiang Wang.
# All rights reserved.

import torch.nn as nn

from fairseq import utils
from fairseq.models import (
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)

from image_captioning.models.text_recognition_encoder import TextRecognitionEncoder
from fairseq.models.lstm import LSTM


@register_model('text_recognition_crnn')
class TextRecognitionCRNNModel(FairseqEncoderDecoderModel):
    """
    CRNN model from `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" (Shi, et al, 2015)
    <https://arxiv.org/abs/1507.05717>`_.

    Args:
        encoder (TextRecognitionEncoder): the encoder
        decoder (CRNNDecoder): the decoder

    The CRNN model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.text_recognition_crnn_parser
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
        parser.add_argument('--backbone', default='densenet_cifar',
                            help='CNN backbone architecture. (default: densenet_cifar)')
        parser.add_argument('--pretrained', action='store_true', help='pretrained')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--no-token-rnn', default=False, action='store_true',
                            help='if set, disables rnn layer')
        parser.add_argument('--no-token-crf', default=False, action='store_true',
                            help='if set, disables conditional random fields')
        parser.add_argument('--decoder-bidirectional', action='store_true',
                            help='make all layers of decoder bidirectional')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)
        encoder = TextRecognitionEncoder(
            args=args,
        )
        decoder = CRNNDecoder(
            dictionary=task.target_dictionary,
            embed_dim=encoder.embed_dim,
            bidirectional=args.decoder_bidirectional,
            num_layers=args.decoder_layers,
            no_token_rnn=args.no_token_rnn,
        )
        return cls(encoder, decoder)

    def forward(self, src_tokens):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source image through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens)
            return self.decoder(encoder_out)

        Args:
            src_tokens (Tensor): tokens in the source image of shape
                `(batch, channel, img_h, img_w)`

        Returns:
            the decoder's output, typically of shape `(tgt_len, batch, vocab)`
        """
        encoder_out = self.encoder(src_tokens)
        decoder_out = self.decoder(encoder_out)

        return decoder_out


class CRNNDecoder(FairseqDecoder):
    """CRNN decoder."""
    def __init__(
        self, dictionary, embed_dim, hidden_size=512,
        bidirectional=True, num_layers=2, no_token_rnn=False,
    ):
        super().__init__(dictionary)
        self.need_rnn = not no_token_rnn
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.rnn = LSTM(
            input_size=embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        ) if self.need_rnn else None

        hidden_size = self.hidden_size if self.need_rnn else embed_dim
        self.classifier = nn.Linear(hidden_size, len(dictionary))

    def forward(self, encoder_out):
        # encoder_out -> decoder
        encoder_out = encoder_out['encoder_out']  # seq_len x bsz x embed_dim

        # get outputs from encoder
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]

        if self.rnn is not None:
            x, _ = self.rnn(encoder_outs, (encoder_hiddens, encoder_cells))
            # Sum bidirectional RNN xputs
            if self.bidirectional:
                x = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:]

        x = self.classifier(x)

        return x

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""
        if log_probs:
            return utils.log_softmax(net_output, dim=2)
        else:
            return utils.softmax(net_output, dim=2)


@register_model_architecture('text_recognition_crnn', 'text_recognition_crnn')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.backbone = getattr(args, 'backbone', 'densenet_cifar')
    args.pretrained = getattr(args, 'pretrained', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.no_token_rnn = getattr(args, 'no_token_rnn', False)
    args.no_token_crf = getattr(args, 'no_token_crf', True)
    args.decoder_bidirectional = getattr(args, 'decoder_bidirectional', False)


@register_model_architecture('text_recognition_crnn', 'decoder_crnn')
def decoder_crnn(args):
    args.decoder_bidirectional = getattr(args, 'decoder_bidirectional', True)
    base_architecture(args)
