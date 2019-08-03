# Copyright (c) 2019-present, Zhiqiang Wang

import math

import torch
import torch.nn as nn

from fairseq.models import FairseqEncoder

from sightseq.modules.features_getter import ConvFeaturesGetter


# output dimensionality for supported architectures
OUTPUT_DIM = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'densenet_cifar': 342,
    'densenet121': 512,
    'densenet161': 2208,
    'densenet169': 1664,
    'densenet201': 1920,
    'mobilenet_v2': 1280,
}


class TextRecognitionEncoder(FairseqEncoder):
    """Text Recognition encoder."""
    def __init__(self, args):
        super(FairseqEncoder, self).__init__()
        self.embed_dim = OUTPUT_DIM[args.backbone]
        self.features = ConvFeaturesGetter(args.backbone, args.pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))

        self.embed_positions = PositionalEncoding(
            embedding_dim=self.embed_dim,
            num_embeddings=self.max_positions(),
        ) if not args.no_token_positional_embeddings else None

        self.decoder_need_rnn = not args.no_token_rnn
        if self.decoder_need_rnn:
            hidden_size = args.decoder_hidden_size

            init_layers = args.decoder_layers
            if args.decoder_bidirectional:
                init_layers *= 2

            self.init_hidden_w = nn.Parameter(
                torch.rand(init_layers, self.embed_dim, hidden_size)
            )  # init_layers x embed_dim x hidden_size
            self.init_hidden_b = nn.Parameter(
                torch.rand(init_layers, 1, hidden_size)
            )  # init_layers x 1 x hidden_size
            self.init_cell_w = nn.Parameter(torch.rand_like(self.init_hidden_w))
            self.init_cell_b = nn.Parameter(torch.rand_like(self.init_hidden_b))

    def forward(self, src_tokens):
        """
        Args:
            src_tokens (Tensor): tokens in the source src_tokens of shape
                `(bsz, channel, img_h, img_w)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, bsz, embed_dim)`
        """
        # src_tokens -> features
        x = self.features(src_tokens)  # bsz x embed_dim x H' x W', where W' stands for `seq_len`
        # features -> pool -> flatten
        x = self.avgpool(x)
        x = x.permute(3, 0, 1, 2).view(x.size(3), x.size(0), -1)  # seq_len x bsz x embed_dim

        final_hiddens, final_cells = None, None

        if self.decoder_need_rnn:
            final_hiddens, final_cells = self.init_hidden(x)

        if self.embed_positions is not None:
            x += self.embed_positions(x)

        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': None,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def init_hidden(self, x):
        mean = torch.mean(x, dim=0)  # bsz x embed_dim

        h0 = mean @ self.init_hidden_w + self.init_hidden_b  # init_layers x bsz x hidden_size
        h0 = torch.tanh(h0)

        c0 = mean @ self.init_cell_w + self.init_cell_b  # init_layers x bsz x hidden_size
        c0 = torch.tanh(c0)

        return (h0, c0)

    def max_positions(self):
        """Maximum sequence length supported by the encoder."""
        return 128


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, embedding_dim, num_embeddings=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(num_embeddings, embedding_dim)  # embed_num x embed_dim
        position = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1)
        emb = math.log(10000.0) / embedding_dim
        emb = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float) * -emb)
        pe[:, 0::2] = torch.sin(position * emb)
        pe[:, 1::2] = torch.cos(position * emb)
        self.register_buffer('pe', pe)

    def forward(self, x):
        out = self.pe[:x.size(0)]  # seq_len x embed_dim
        out = out.unsqueeze(1)
        out = out.expand_as(x)
        return out
