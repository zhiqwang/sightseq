"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018).
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import math

__all__ = ['MobileNetV2', 'mobilenetv2_cifar']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self, inplanes, outplanes, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidplanes = round(inplanes * expand_ratio)
        self.identity = stride == 1 and inplanes == outplanes

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # depthwise conv
                nn.Conv2d(hidplanes, hidplanes, 3, stride, 1, groups=hidplanes, bias=False),
                nn.BatchNorm2d(hidplanes),
                nn.ReLU6(inplace=True),
                # pointwise conv linear
                nn.Conv2d(hidplanes, outplanes, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outplanes),
            )
        else:
            self.conv = nn.Sequential(
                # pointwise conv
                nn.Conv2d(inplanes, hidplanes, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidplanes),
                nn.ReLU6(inplace=True),
                # depthwise conv
                nn.Conv2d(hidplanes, hidplanes, 3, stride, 1, groups=hidplanes, bias=False),
                nn.BatchNorm2d(hidplanes),
                nn.ReLU6(inplace=True),
                # pointwise conv linear
                nn.Conv2d(hidplanes, outplanes, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outplanes),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, block, num_classes=10, width_mult=1., small_inputs=True):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # expansion, planes, num_blocks, stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        if small_inputs:
            self.cfgs[1] = [6, 24, 2, 1]  # NOTE: change stride 2 -> 1 for CIFAR10
        # First convolution
        num_init_features = _make_divisible(32 * width_mult, 8)
        stride = 1 if small_inputs else 2  # NOTE: change stride 2 -> 1 for CIFAR10
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, 3, stride, 1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU6(inplace=True)),
        ]))

        inplanes = num_init_features
        # building inverted residual blocks
        for i, (expansion, planes, num_blocks, stride) in enumerate(self.cfgs):
            outplanes = _make_divisible(planes * width_mult, 8)
            self.features.add_module('invertedresidual{}_0'.format(i), block(inplanes, outplanes, stride, expansion))
            inplanes = outplanes
            for n in range(1, num_blocks):
                self.features.add_module('invertedresidual{}_{}'.format(i, n), block(inplanes, outplanes, 1, expansion))
                inplanes = outplanes

        # building last convolution
        outplanes = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.features.add_module('conv_final', nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False))
        self.features.add_module('norm_final', nn.BatchNorm2d(outplanes))
        self.features.add_module('relu_final', nn.ReLU6(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Linear layer
        self.classifier = nn.Linear(outplanes, num_classes)

        self._initialize_weights()

    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2_cifar(pretrained=False, **kwargs):
    model = MobileNetV2(InvertedResidual, **kwargs)
    return model


if __name__ == '__main__':

    model = mobilenetv2_cifar()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(y.size())
