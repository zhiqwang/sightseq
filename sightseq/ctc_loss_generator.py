# Copyright (c) 2019-present, Zhiqiang Wang

import math

import torch
from fairseq.sequence_generator import EnsembleModel


class CTCLossGenerator(object):
    """Scores the target for a given source image."""

    def __init__(
        self,
        tgt_dict,
        retain_dropout=False,
        raw=False,
        strings=False
    ):
        self.tgt_dict = tgt_dict
        self.blank_idx = tgt_dict.blank()
        self.retain_dropout = retain_dropout
        self.raw = raw
        self.strings = strings

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of images with best path decoding."""
        model = CTCLossEnsembleModel(models)
        if not self.retain_dropout:
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }

        encoder_outs = model.forward_encoder(encoder_input)
        log_probs = model.forward_decoder(encoder_outs)
        hypos = []
        lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.int32)
        _, decoder_out = log_probs.max(2)
        decoder_out = decoder_out.transpose(1, 0).contiguous().reshape(-1)  # reshape
        tokens = self.decode(decoder_out, lengths)
        if isinstance(tokens[0], list):
            for token in tokens:
                hypos.append([{
                    'tokens': token,
                    'score': None,
                    'attention': None,
                    'alignment': None,
                    'positional_scores': None,
                }])
        else:
            hypos = [[{
                'tokens': tokens,
                'score': None,
                'attention': None,
                'alignment': None,
                'positional_scores': None,
            }]]

        return hypos

    def decode(self, decoder_out, length):
        """Decode encoded labels back into strings.
        Args:
            decoder_out: torch.IntTensor [length_0 + length_1 + ...
                length_{n - 1}]: encoded labels.
            lenght: torch.IntTensor [n]: length of each labels.
        Raises:
            AssertionError: when the labels and its length does not match.
        Returns:
            labels (str or list of str): labels to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert decoder_out.numel() == length
            if self.raw:
                if self.strings:
                    return u''.join([self.tgt_dict.symbols[i] for i in decoder_out]).encode('utf-8')
                return decoder_out.tolist()
            else:
                decoder_out_non_blank = []
                for i in range(length):  # removing repeated characters and blank.
                    if (decoder_out[i] != self.blank_idx and (not (i > 0 and decoder_out[i - 1] == decoder_out[i]))):
                        if self.strings:
                            decoder_out_non_blank.append(self.tgt_dict.symbols[decoder_out[i]])
                        else:
                            decoder_out_non_blank.append(decoder_out[i].item())
                if self.strings:
                    return u''.join(decoder_out_non_blank).encode('utf-8')
                return decoder_out_non_blank
        else:  # batch mode
            assert decoder_out.numel() == length.sum()
            labels = []
            index = 0
            for i in range(length.numel()):
                idx_end = length[i]
                labels.append(self.decode(decoder_out[index:index + idx_end], torch.IntTensor([idx_end])))
                index += idx_end
            return labels


class CTCLossEnsembleModel(EnsembleModel):
    def __init__(self, models):
        super().__init__(models)

    @torch.no_grad()
    def forward_decoder(self, encoder_outs, temperature=1.):
        if len(self.models) == 1:
            return self._decode_one(
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                log_probs=True,
                temperature=temperature,
            )

        log_probs = []
        for model, encoder_out in zip(self.models, encoder_outs):
            probs = self._decode_one(
                model,
                encoder_out,
                log_probs=True,
                temperature=temperature,
            )
            log_probs.append(probs)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        return avg_probs

    def _decode_one(
        self, model, encoder_out, log_probs,
        temperature=1.,
    ):
        decoder_out = model.decoder(encoder_out)
        if temperature != 1.:
            decoder_out.div_(temperature)
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        return probs
