#!/usr/bin/env python3 -u
# Copyright (c) 2019-present, Zhiqiang Wang

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from fairseq import tasks


def from_pretrained(
    model_name_or_path,
    checkpoint_file='model.pt',
    data_name_or_path='.',
    archive_map=None,
    **kwargs,
):

    if archive_map is not None:
        if model_name_or_path in archive_map:
            model_name_or_path = archive_map[model_name_or_path]
        if data_name_or_path is not None and data_name_or_path in archive_map:
            data_name_or_path = archive_map[data_name_or_path]

    args = kwargs['args']
    task = tasks.setup_task(args)
    model = task.build_model(args)

    state_dict = load_state_dict_from_url(model_name_or_path)
    model.load_state_dict(state_dict, strict=True)

    return {
        'args': args,
        'task': task,
        'model': model,
    }
