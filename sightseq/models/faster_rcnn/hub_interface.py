# Copyright (c) 2019-present, Zhiqiang Wang

import torch
import torch.nn as nn


class FasterRCNNHubInterface(nn.Module):
    """A simple PyTorch Hub interface to FasterRCNN.

    Usage: https://github.com/zhiqwang/sightseq/tree/master/examples/object_detection
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model

        # this is useful for determining the device
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device
