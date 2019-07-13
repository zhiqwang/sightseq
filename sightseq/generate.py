#!/usr/bin/env python3 -u

# Copyright (c) 2019-present, Zhiqiang Wang.
# Modified from https://github.com/pytorch/fairseq

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Recognize pre-processed image with a trained model.
"""

import collections

import torch

from fairseq import bleu, options, tasks, checkpoint_utils, progress_bar, utils
from fairseq.meters import StopwatchMeter, TimeMeter


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu
    use_ctc_loss = True if args.criterion == 'ctc_loss' else False

    # Setup task, e.g., image captioning
    task = tasks.setup_task(args)
    # Load dataset split
    task.load_dataset(args.gen_subset, combine=True, epoch=0)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    model_paths = args.path.split(':')
    models, _model_args = checkpoint_utils.load_model_ensemble(
        model_paths,
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())

    stats = collections.OrderedDict()
    num_sentences = 0
    num_correct = 0
    has_target = True

    with progress_bar.build_progress_bar(
        args, itr,
        prefix='inference on \'{}\' subset'.format(args.gen_subset),
        no_progress_bar='simple',
    ) as progress:
        wps_meter = TimeMeter()
        for sample in progress:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            gen_timer.start()
            hypos = task.inference_step(generator, models, sample)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None
                target_tokens = None
                if has_target:
                    if use_ctc_loss:
                        target_tokens = sample['target'][i]
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)
                    else:
                        # Remove padding
                        target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()
                        # Regenerate original sentences from tokens.
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                if not args.quiet:
                    if has_target:
                        print('\nT-{}\t{}'.format(sample_id, target_str))

                # Process top predictions
                hypo = hypos[i][0]
                hypo_tokens = hypo['tokens'] if use_ctc_loss else hypo['tokens'].int().cpu()
                hypo_str = tgt_dict.string(hypo_tokens, args.remove_bpe, escape_unk=True)
                alignment = hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None

                if hypo_str == target_str:
                    num_correct += 1

                if not args.quiet:
                    print('H-{}\t{}\t{}'.format(sample_id, hypo_str, hypo['score']))
                    print('P-{}\t{}'.format(
                        sample_id,
                        ' '.join(map(
                            lambda x: '{:.4f}'.format(x),
                            hypo['positional_scores'].tolist(),
                        )) if not use_ctc_loss else None
                    ))

                    if args.print_alignment:
                        print('A-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(lambda x: str(utils.item(x)), alignment))
                        ))

                # Score only the top hypothesis
                if has_target:
                    if hasattr(scorer, 'add_string'):
                        scorer.add_string(target_str, hypo_str)
                    else:
                        scorer.add(target_tokens, hypo_tokens)

            wps_meter.update(num_generated_tokens)
            num_sentences += sample['nsentences']
            stats['wps'] = round(wps_meter.avg)
            stats['acc'] = num_correct / num_sentences
            progress.log(stats, tag='accuracy')
        progress.print(stats, tag='accuracy')

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))
    return scorer


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
