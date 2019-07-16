# Copyright (c) 2019-present, Zhiqiang Wang.

import torchvision.transforms as transforms

from fairseq.data.transforms import register_transform


@register_transform('image')
class ImageTransform(object):

    def __init__(self, source_lang=None, target_lang=None):

        image_size = self.args.height if self.args.keep_ratio else (self.args.height, self.args.width)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.396, 0.576, 0.562], std=[0.154, 0.128, 0.130]),
        ])

    def encode(self, x: str) -> str:
        return ' '.join(self.transform(x))

    def decode(self, x: str) -> str:
        return x
