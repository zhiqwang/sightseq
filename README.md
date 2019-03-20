# Convolutional Recurrent Neural Network

This software implements the Convolutional Recurrent Neural Network (CRNN) in pytorch in paper:

**An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition**,
Baoguang Shi, Xiang Bai, Cong Yao,
PAMI 2017 [[arXiv](https://arxiv.org/abs/1507.05717)]

## What is it?

This code implements: (args.arch)

1. DenseNet + CTCLoss (densenet_cifar, densenet121)
2. ResNet + CTCLoss (resnet_cifar)
3. MobileNetV2 + CTCLoss (mobilenetv2_cifar)
4. ShuffleNetV2 + CTCLoss (shufflenetv2_cifar)

## Prerequisites

In order to run this toolbox you will need:

- Python3 (tested with Python 3.6+)
- PyTorch deep learning framework (tested with version 1.0.1)

## Demo

The demo reads an example image and recognizes its text content. See the [demo notebook](./demo.ipynb) for all the details.

Example image:

![Example Image](./test/54439593_2298493320.jpg)

Expected output:

    -停--下--来--，--看--着--那--些--握--着------ => 停下来，看着那些握着

## Usage

- Navigate (`cd`) to the root of the toolbox `[YOUR_CRNN_ROOT]`.
- Resize the height of a image to 32, and the width should be divisible by 8.

### Datasets

Refer to YCG09's [SynthText](https://github.com/YCG09/chinese_ocr), the image size is 32x280, origin image can be downloaded from [BaiduYun](https://pan.baidu.com/s/1QkI7kjah8SPHwOQ40rS1Pw) (pw: lu7m), untar it to directory `[DATASET_ROOT_DIR]`.

### Annotation file format

In each line in the annotation file, the format is:
    
    img_path encode1 encode2 encode3 encode4 encode5 ...

where `encode` is the sequence's encode.

### Alphabet

Altogether 5989 characters, containing Chinese characters, English letters, numbers and punctuation, can be downloaded from [OneDrive](https://1drv.ms/t/s!AtlbCejIR3IcgQjX2JYMSC0tEcpx) or [BaiduYun](https://pan.baidu.com/s/1XCUBTtWx9K6fgQeINjCK-g) (pw: d654), put the downloaded file `alphabet_decode_5990.txt` into directory `[DATASET_ROOT_DIR]`.

### Pretrained Model

Training with `densenet121` architecture and pre-trained models can be found [OneDrive](https://1drv.ms/u/s!AtlbCejIR3IcgQkwuQkN1aAoPHX8) or [BaiduYun](https://pan.baidu.com/s/163fBRV6S8WgwImPHnee_gg) (pw: riuh). P.S. current pretrained model is rough, I hope that I have time to modify it later.

### Training

Training strategy:

    python ./main.py --arch densenet121 --alphabet [DATASET_ROOT_DIR]/alphabet_decode_5990.txt --dataset-root [DATASET_ROOT_DIR] --lr 5e-5 --optimizer rmsprop --gpu-id 0 --not-pretrained

### Testing

Use trained model to test:

    python ./main.py --arch densenet121 --alphabet [DATASET_ROOT_DIR]/alphabet_decode_5990.txt --dataset-root [DATASET_ROOT_DIR] --lr 5e-5 --optimizer rmsprop --gpu-id 0 --resume densenet121_pretrained.pth.tar --test-only

## Reference
- [crnn.pytorch](https://github.com/meijieru/crnn.pytorch)
- [CIFAR10 with PyTorch](https://github.com/kuangliu/pytorch-cifar)
- [Efficient densenet pytorch](https://github.com/gpleiss/efficient_densenet_pytorch)
