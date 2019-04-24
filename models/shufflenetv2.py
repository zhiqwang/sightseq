'''ShuffleNetV2 in PyTorch.

See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ShuffleNetV2', 'shufflenetv2_cifar']


class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N, C, H, W] -> [N, g, C/g, H, W] -> [N, C / g, g, H, w] -> [N, C, H, W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class DownsampleBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(DownsampleBlock, self).__init__()
        outplanes = planes // 2
        # left
        self.conv1l = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, groups=inplanes, bias=False)
        self.bn1l = nn.BatchNorm2d(inplanes)
        self.conv2l = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.bn2l = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        # right
        self.conv1r = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.bn1r = nn.BatchNorm2d(outplanes)
        self.conv2r = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=2, padding=1, groups=outplanes, bias=False)
        self.bn2r = nn.BatchNorm2d(outplanes)
        self.conv3r = nn.Conv2d(outplanes, outplanes, kernel_size=1, bias=False)
        self.bn3r = nn.BatchNorm2d(outplanes)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        # left
        out_l = self.bn1l(self.conv1l(x))
        out_l = self.relu(self.bn2l(self.conv2l(out_l)))
        # right
        out_r = self.relu(self.bn1r(self.conv1r(x)))
        out_r = self.bn2r(self.conv2r(out_r))
        out_r = self.relu(self.bn3r(self.conv3r(out_r)))
        # concat
        out = torch.cat([out_l, out_r], 1)
        out = self.shuffle(out)
        return out


class InvertedResidual(nn.Module):
    def __init__(self, inplanes, split_ratio=0.5):
        super(InvertedResidual, self).__init__()
        self.split = SplitBlock(split_ratio)
        inplanes = int(inplanes * split_ratio)
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, groups=inplanes, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.conv3 = nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = self.relu(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        out = self.relu(self.bn3(self.conv3(out)))
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out


class ShuffleNetV2(nn.Module):

    def __init__(self, downsample_block, basic_block, width_mult=1.0, num_classes=10, small_inputs=True):
        super(ShuffleNetV2, self).__init__()
        self.small_inputs = small_inputs
        self.cfg = {
            0.5: (48, 96, 192, 1024),
            1.0: (116, 232, 464, 1024),
            1.5: (176, 352, 704, 1024),
            2.0: (224, 488, 976, 2048),
        }

        stage_out_planes = self.cfg[width_mult]
        num_stages = (3, 7, 3)

        self.downsample_block = downsample_block
        self.basic_block = basic_block
        self.inplanes = 24
        outplanes = stage_out_planes[3]
        stride = 1 if small_inputs else 2  # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(stage_out_planes[0], num_stages[0])
        self.layer2 = self._make_layer(stage_out_planes[1], num_stages[1])
        self.layer3 = self._make_layer(stage_out_planes[2], num_stages[2])
        self.conv2 = nn.Conv2d(stage_out_planes[2], outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(outplanes, num_classes)

    def _make_layer(self, planes, stages):
        layers = [self.downsample_block(self.inplanes, planes)]
        for i in range(stages):
            layers.append(self.basic_block(planes))
        self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if not self.small_inputs:  # NOTE: Undo MaxPool for CIFAR10
            out = F.max_pool2d(out, 3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def shufflenetv2_cifar(pretrained=False, **kwargs):
    model = ShuffleNetV2(DownsampleBlock, InvertedResidual, width_mult=1.0, **kwargs)
    return model


if __name__ == '__main__':

    model = shufflenetv2_cifar(small_inputs=True)
    x = torch.randn(16, 3, 32, 32)
    y = model(x)
    print(y.size())
