# Copyright (c) 2019-present, Zhiqiang Wang.
# All rights reserved.

import os
import torchvision.transforms as transforms

from fairseq.tasks import FairseqTask, register_task
from fairseq.data import Dictionary
from image_captioning.data import CTCLossDictionary, TextRecognitionDataset


@register_task('text_recognition')
class TextRecognitionTask(FairseqTask):
    """
    Train a text recognition model.

    Args:
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target text
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--max-positions', default=1024, type=int,
                            help='max input length')
        parser.add_argument('--height', type=int, default=32,
                            help='image height size used for training (default: 32)')
        parser.add_argument('--width', type=int, default=200,
                            help='image width size used for training (default: 200)')
        parser.add_argument('--keep-ratio', action='store_true',
                            help='keep image size ratio when training')
        parser.add_argument('--no-token-pin-memory', default=False, action='store_true',
                            help='training using pined memory')
        # fmt: on

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        use_ctc_loss = True if args.criterion == 'ctc_loss' else False
        tgt_dict = cls.load_dictionary(os.path.join(args.data, 'dict.txt'), use_ctc_loss)
        print('| target dictionary: {} types'.format(len(tgt_dict)))

        return cls(args, tgt_dict)

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict

    @classmethod
    def load_dictionary(cls, filename, use_ctc_loss):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        if use_ctc_loss:
            return CTCLossDictionary.load(filename)
        return Dictionary.load(filename)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # Read input images and targets
        image_names = []
        targets = []
        target_lengths = []
        image_root = os.path.join(self.args.data, 'images')
        label_path = os.path.join(self.args.data, '{}.txt'.format(split))
        with open(label_path, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split()
                image_names.append(os.path.join(image_root, line[0]))
                targets.append(line[1:])
                target_lengths.append(len(line[1:]))

        assert len(image_names) == len(targets) == len(target_lengths)
        print('| {} {} {} images'.format(self.args.data, split, len(image_names)))

        mean = [0.396, 0.576, 0.562]
        std = [0.154, 0.128, 0.130]

        image_size = self.args.height if self.args.keep_ratio else (self.args.height, self.args.width)

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        shuffle = True if split == 'train' else False
        append_eos_to_target = False if self.args.criterion == 'ctc_loss' else True
        use_ctc_loss = True if self.args.criterion == 'ctc_loss' else False
        self.datasets[split] = TextRecognitionDataset(
            image_names, targets, self.tgt_dict, tgt_sizes=target_lengths,
            shuffle=shuffle, transform=transform, use_ctc_loss=use_ctc_loss,
            input_feeding=True, append_eos_to_target=append_eos_to_target,
        )

    def build_generator(self, args):
        if args.criterion == 'ctc_loss':
            from image_captioning.ctc_loss_generator import CTCLossGenerator
            return CTCLossGenerator(self.target_dictionary)
        else:
            from fairseq.sequence_generator import SequenceGenerator
            return SequenceGenerator(
                self.target_dictionary,
                beam_size=args.beam,
                max_len_a=args.max_len_a,
                max_len_b=args.max_len_b,
                min_len=args.min_len,
                stop_early=(not args.no_early_stop),
                normalize_scores=(not args.unnormalized),
                len_penalty=args.lenpen,
                unk_penalty=args.unkpen,
                sampling=args.sampling,
                sampling_topk=args.sampling_topk,
                temperature=args.temperature,
                diverse_beam_groups=args.diverse_beam_groups,
                diverse_beam_strength=args.diverse_beam_strength,
                match_source_len=args.match_source_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and
        # In order to use `CuDNN`, the "target" has max length 256,
        return (self.args.max_positions, 256)

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
