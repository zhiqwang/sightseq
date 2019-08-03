# Copyright (c) 2019-present, Zhiqiang Wang

import torch.nn as nn
import sightseq.modules as modules


class ConvFeaturesGetter(nn.Module):
    """CNN features getter for the Encoder of image."""
    def __init__(self, backbone_name, pretrained):
        super().__init__()
        # loading network
        conv_model_in = getattr(modules, backbone_name)(pretrained=pretrained)

        if backbone_name.startswith('resnet') or backbone_name.startswith('mobilenet'):
            conv = list(conv_model_in.children())[:-2]
        elif backbone_name.startswith('densenet'):
            conv = list(conv_model_in.features.children())
            conv.append(nn.ReLU(inplace=True))
        else:
            raise ValueError('Unsupported or unknown architecture: {}!'.format(backbone_name))

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
