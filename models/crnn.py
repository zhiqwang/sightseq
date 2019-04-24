import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import models

# pretrained features
FEATURES = {}

# output dimensionality for supported architectures
OUTPUT_DIM = {
    'resnet_cifar': 512,
    'densenet_cifar': 342,
    'densenet121': 384,
    'mobilenetv2_cifar': 1280,
    'shufflenetv2_cifar': 1024,
}


class CRNN(nn.Module):

    def __init__(self, features, meta):

        super(CRNN, self).__init__()
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        self.classifier = nn.Linear(meta['output_dim'], meta['num_classes'])

        self.meta = meta

    def forward(self, x):
        # x -> features
        out = self.features(x)
        # features -> pool -> flatten -> decoder -> softmax
        out = self.avgpool(out)
        out = out.permute(3, 0, 1, 2).view(out.size(3), out.size(0), -1)
        out = self.classifier(out)
        out = F.log_softmax(out, dim=2)

        return out

    def __repr__(self):
        tmpstr = super(CRNN, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n'  # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     output dim: {}\n'.format(self.meta['output_dim'])
        tmpstr += '     classes: {}\n'.format(self.meta['num_classes'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr


def init_network(params):

    # parse params with default values
    architecture = params.get('architecture', 'densenet_cifar')
    num_classes = params.get('num_classes', 11)
    mean = params.get('mean', [0.396, 0.576, 0.562])
    std = params.get('std', [0.154, 0.128, 0.130])
    pretrained = params.get('pretrained', False)

    # get output dimensionality size
    dim = OUTPUT_DIM[architecture]

    # loading network
    if pretrained:
        if architecture not in FEATURES:
            # initialize with network pretrained on imagenet in pytorch
            net_in = getattr(models, architecture)(pretrained=True)
        else:
            # initialize with random weights, later on we will fill features with custom pretrained network
            net_in = getattr(models, architecture)(pretrained=False)
    else:
        # initialize with random weights
        net_in = getattr(models, architecture)(pretrained=False)

    # initialize features
    # take only convolutions for features,
    # always ends with ReLU to make last activations non-negative
    if architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif architecture.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif architecture.startswith('mobilenetv2'):
        features = list(net_in.children())[:-2]
    elif architecture.startswith('shufflenetv2'):
        features = list(net_in.children())[:-2]
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    # create meta information to be stored in the network
    meta = {
        'architecture': architecture,
        'num_classes': num_classes,
        'mean': mean,
        'std': std,
        'output_dim': dim,
    }

    # create a generic crnn network
    net = CRNN(features, meta)

    # initialize features with custom pretrained network if needed
    if pretrained and architecture in FEATURES:
        print(">> {}: for '{}' custom pretrained features '{}' are used".format(
              os.path.basename(__file__), architecture, os.path.basename(FEATURES[architecture])))
        net.features.load_state_dict(torch.load(FEATURES[architecture]))

    return net
