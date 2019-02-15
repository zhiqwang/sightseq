"""
@Title: DenseNets on PyTorch for CIFAR-10
@References:
[1] Gao Huang, Zhuang Liu, Laurens van der Maaten
    Densely Connected Deep Convolutional Networks. arXiv:1512.03385
    
[2] PyTorch Open Source Repository
    https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py   
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def flatten(x):
    return x.permute(3, 0, 1, 2).reshape(x.shape[3], x.shape[0], -1)


class _Bottleneck(nn.Module):
    '''
    Before entering the first dense block, a convolution with 16 (or twice the 
    growth rate for BC type) output channels is performed on the input images
    '''
    def __init__(self, num_input_features, bn_size, growth_rate):
        super(_Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class _Transition(nn.Module):
    
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()

        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.avgpool(out)
        return out


class DenseNet(nn.Module):

    def __init__(self, growth_rate=12, depth=100, bn_size=4,
                 reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()

        compression = True if reduction < 1 else False  # Determine if DenseNet-C

        num_layers = (depth - 4) // 6
        num_input_features = 2 * growth_rate if compression else 16

        # First convolution
        self.conv1 = nn.Conv2d(3, num_input_features, kernel_size=3, padding=1, bias=False)

        # Dense Block 1 
        self.dense1 = self._make_dense(num_input_features, bn_size, growth_rate, num_layers)
        num_input_features += num_layers * growth_rate
        num_output_features = int(math.floor(num_input_features * reduction))

        # _Transition Block 1
        self.trans1 = _Transition(num_input_features, num_output_features)
        num_input_features = num_output_features

        # Dense Block 2
        self.dense2 = self._make_dense(num_input_features, bn_size, growth_rate, num_layers)
        num_input_features += num_layers * growth_rate
        num_output_features = int(math.floor(num_input_features * reduction))

        # _Transition Block 2
        self.trans2 = _Transition(num_input_features, num_output_features)
        num_input_features = num_output_features

        # Dense Block 3
        self.dense3 = self._make_dense(num_input_features, bn_size, growth_rate, num_layers)

        # _Transition Block 3
        num_input_features += num_layers * growth_rate
        self.bn1 = nn.BatchNorm2d(num_input_features)

        # Dense Layer
        self.avgpool = nn.AvgPool2d(kernel_size=(8,1))

        self.flatten = flatten
        self.fc = nn.Linear(num_input_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_dense(self, num_input_features, bn_size, growth_rate, num_layers):
        ''' Function to build a Dense Block '''
        layers = []
        for i in range(int(num_layers)):
            layers.append(_Bottleneck(num_input_features, bn_size, growth_rate))
            num_input_features += growth_rate
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)                     # 32x204
        out = self.trans1(self.dense1(out))     # 16x102
        out = self.trans2(self.dense2(out))     # 8x51
        out = self.dense3(out)                  # 8x51
        out = self.avgpool(out)                 # 1x51
        out = self.flatten(out)
        out = self.fc(out)
        out = F.log_softmax(out, dim=2)
        return out


def denseNetBC_100_12(num_classes=10):
    return DenseNet(depth=100, growth_rate=12, reduction=0.5, num_classes=num_classes)

if __name__ == '__main__':

    model = denseNetBC_100_12(num_classes=11)
    y = model(torch.randn(1, 3, 32, 204))
    print(y.shape)
