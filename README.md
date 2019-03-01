## Convolutional Recurrent Neural Network

This software implements the Convolutional Recurrent Neural Network (CRNN) in pytorch in paper:

**An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition**,
Baoguang Shi, Xiang Bai, Cong Yao,
PAMI 2017 [[arXiv](https://arxiv.org/abs/1507.05717)]

### What is it?

This code implements: (args.arch)

1. DenseNet + CTCLoss (densenet_cifar)
2. ResNet + CTCLoss (resnet_cifar)
3. MobileNetV2 + CTCLoss (mobilenetv2_cifar)
4. ShuffleNetV2 + CTCLoss (shufflenetv2_cifar)

### Prerequisites

In order to run this toolbox you will need:

- Python3 (tested with Python 3.7.0)
- PyTorch deep learning framework (tested with version 1.0.1)

### Usage

- Navigate (`cd`) to the root of the toolbox `[YOUR_CRNN_ROOT]`.
- Resize the height of a image to 32, and the width should be divisible by 8.

#### Training

Training strategy:
```
python ./main.py --not-pretrained --arch densenet_cifar
```

#### Testing

Launch the demo by:
```
python ./main.py --test-only --resume densenet_cifar_best.pth.tar
```

### Reference
- [crnn.pytorch](https://github.com/meijieru/crnn.pytorch)
- [CIFAR10 with PyTorch](https://github.com/kuangliu/pytorch-cifar)
- [Efficient densenet pytorch](https://github.com/gpleiss/efficient_densenet_pytorch)
