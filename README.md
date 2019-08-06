# 🔭sightseq

Now, Let's go **sight**seeing by vision and **seq**uence language multimodal around the deep learning world.

### What's New:

- **July 30, 2019:** Add faster rcnn models. And I rename this repo from *image-captioning* to *sightseq*, this is the last time I rename this repo, I promise.
- **June 11, 2019:** I rewrite the text recognition part base on [fairseq](https://github.com/pytorch/fairseq). Stable version refer to branch [crnn](https://github.com/zhiqwang/image-captioning/tree/crnn), which provides pre-trained model checkpoints. Current branch is work in process. Very pleasure for suggestion and cooperation in the fairseq text recognition project.

### Features:
sightseq provides reference implementations of various deep learning tasks, including:

- **Text Recognition**
  - [Shi et al. (2015), CRNN: An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)
- **Object Detection**
  - **_New_** [Ren et al. (2015), Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

**Additionally:**
- All features of fairseq
- Flexible to enable convolution layer, recurrent layer in CRNN
- Positional Encoding of images

# General Requirements and Installation
- [PyTorch](http://pytorch.org/) (There is a [bug](https://github.com/pytorch/pytorch/pull/21244) in [nn.CTCLoss](https://pytorch.org/docs/master/nn.html#ctcloss) which is solved in nightly version)
- Python version >= 3.5
- [Fairseq](https://github.com/pytorch/fairseq) version >= 0.7.1
- [torchvision](https://github.com/pytorch/vision) version >= 0.3.0
- For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

# Pre-trained models and examples
- [text recognition](examples/text_recognition)
- [object detection](examples/object_detection)

# License
sightseq is MIT-licensed.
The license applies to the pre-trained models as well.
