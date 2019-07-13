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
Data pre-processing: build vocabularies and binarize training data.
"""

import os
from fairseq import options, tasks, utils


def main(args):
    utils.import_user_module(args)

    print(args)

    os.makedirs(args.destdir, exist_ok=True)

    task = tasks.get_task(args.task)

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_dictionary(filenames, src=False, tgt=False):
        assert src ^ tgt
        return task.build_dictionary(
            filenames,
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
        )

    if not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
        raise FileExistsError(dict_path(args.target_lang))

    if args.tgtdict:
        tgt_dict = task.load_dictionary(args.tgtdict)
    else:
        assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
        tgt_dict = build_dictionary([train_path(args.target_lang)], tgt=True)

    if tgt_dict is not None:
        tgt_dict.save(dict_path(args.target_lang))

    print("| Wrote preprocessed data to {}".format(args.destdir))


def cli_main():
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
