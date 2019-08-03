# Copyright (c) 2019-present, Zhiqiang Wang

import argparse
import importlib
import os

from sightseq.models.text_recognition_encoder import TextRecognitionEncoder

MODEL_REGISTRY = {}
ARCH_MODEL_INV_REGISTRY = {}


__all__ = [
    'TextRecognitionEncoder',
]


# automatically import any Python files in the models/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        model_name = file[:file.find('.py')]
        module = importlib.import_module('sightseq.models.' + model_name)

        # extra `model_parser` for sphinx
        if model_name in MODEL_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_archs = parser.add_argument_group('Named architectures')
            group_archs.add_argument('--arch', choices=ARCH_MODEL_INV_REGISTRY[model_name])
            group_args = parser.add_argument_group('Additional command-line arguments')
            MODEL_REGISTRY[model_name].add_args(group_args)
            globals()[model_name + '_parser'] = parser
