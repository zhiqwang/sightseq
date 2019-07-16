# Copyright (c) 2019-present, Zhiqiang Wang.

import importlib
import os

from fairseq import registry


build_transform, register_transform, TRANSFORM_REGISTRY = registry.setup_registry(
    '--transform',
    default='space',
)


# automatically import any Python files in the transforms/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('sightseq.data.transforms.' + module)
