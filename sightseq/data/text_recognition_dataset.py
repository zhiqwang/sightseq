# Copyright (c) 2019-present, Zhiqiang Wang

import numpy as np
import torch
from fairseq.data import data_utils, FairseqDataset

from sightseq.data.data_utils import default_loader


def collate(
    samples, pad_idx, eos_idx, left_pad=False,
    input_feeding=True, use_ctc_loss=False,
):
    """collate samples of images and targets."""
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    images = torch.stack([s['source'] for s in samples])

    prev_output_tokens = None

    if use_ctc_loss:
        targets = [s['target'] for s in samples]
    else:
        targets = merge('target', left_pad=left_pad)
        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad,
                move_eos_to_beginning=True,
            )

    tgt_lengths = [s['target_length'] for s in samples]
    id = torch.LongTensor([s['id'] for s in samples])
    ntokens = sum(tgt_lengths)
    tgt_lengths = torch.IntTensor(tgt_lengths)

    # TODO: pin-memory
    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': images,
        },
        'target': targets,
        'target_length': tgt_lengths,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class TextRecognitionDataset(FairseqDataset):
    """A dataset that provides helpers for batching."""

    def __init__(
        self, src, tgt, tgt_dict, tgt_sizes=None,
        shuffle=True, transform=None, loader=default_loader,
        use_ctc_loss=False, left_pad=False,
        input_feeding=True, append_eos_to_target=False,
    ):
        self.src = src
        self.tgt = tgt
        self.tgt_dict = tgt_dict
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.transform = transform
        self.loader = loader
        self.shuffle = shuffle
        self.use_ctc_loss = use_ctc_loss
        self.left_pad = left_pad
        self.input_feeding = input_feeding
        self.append_eos_to_target = append_eos_to_target

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        image_name = self.src[index]
        image = self.loader(image_name)

        if self.transform is not None:
            image = self.transform(image)

        tgt_item = self.tgt[index]
        tgt_length = self.tgt_sizes[index]
        # Convert label to a numeric ID.
        tgt_item = torch.IntTensor([self.tgt_dict.index(i) for i in tgt_item])
        # Append EOS to end of tgt sentence if it does not have an EOS
        if self.append_eos_to_target:
            tgt_item = tgt_item.to(torch.int64)
            eos = self.tgt_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([tgt_item, torch.LongTensor([eos])])

        return {
            'id': index,
            'source': image,
            'target': tgt_item,
            'target_length': tgt_length,
        }

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        pad_idx = None if self.use_ctc_loss is True else self.tgt_dict.pad()
        eos_idx = None if self.use_ctc_loss is True else self.tgt_dict.eos()
        left_pad = self.left_pad
        input_feeding = self.input_feeding

        return collate(
            samples, pad_idx=pad_idx, eos_idx=eos_idx, left_pad=left_pad,
            input_feeding=input_feeding, use_ctc_loss=self.use_ctc_loss,
        )

    def ordered_indices(self):
        """
        Return an ordered list of indices. Batches will be constructed based
        on this order.
        """
        if self.shuffle:
            return np.random.permutation(len(self))
        else:
            return np.arange(len(self))

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.tgt_sizes[index] if self.tgt_sizes is not None else 0

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.tgt_sizes[index] if self.tgt_sizes is not None else 0
