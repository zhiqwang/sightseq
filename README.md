# Introduction

Now, Let's go **sight**seeing by vision and **seq**uence language multimodal around the deep learning world. Sightseq provides reference implementations of various image captioning models, including:
- **Convolutional Recurrent Neural Network (CRNN)**
  - [Shi et al. (2015): An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)
- **Attention networks**
- **Transformer networks**

Sightseq features:

- All features of Fairseq
- Flexible to enable convolution layer, recurrent layer in CRNN
- Positional Encoding of images

## Updates

**June 11, 2019:** I rewrite this text recognition repo base on [fairseq](https://github.com/pytorch/fairseq). Stable version refer to branch [crnn](https://github.com/zhiqwang/image-captioning/tree/crnn), which provides pre-trained model checkpoints. Current branch is under construction. Very pleasure for suggestion and cooperation in the fairseq text recognition project.

## Requirements and Installation

* [PyTorch](http://pytorch.org/) (There is a [bug](https://github.com/pytorch/pytorch/pull/21244) in [nn.CTCLoss](https://pytorch.org/docs/master/nn.html#ctcloss) which is solved in nightly version)
* Python version >= 3.6
* [Fairseq](https://github.com/pytorch/fairseq) version >= 0.7.1
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

## Usage

- Navigate (`cd`) to the root of the toolbox `[YOUR_sightseq_ROOT]`.

## Annotation file format

In each line in the annotation file, the format is:

    img_path char1 char2 char3 char4 char5 ...

where the `char` is the sequence's character.

For example, there is task identifying numbers of an image, the `Alphabet` is "0123456789". And there is an image named "00120_00091.jpg" in folder `[DATA]/images`, its constant is "99353361056742", there should be a line in the `[DATA]/train.txt` or `[DATA]/valid.txt`.

    00120_00091.jpg 9 9 3 5 3 3 6 1 0 5 6 7 4 2

## Preprocess

Generate `dict.txt` strategy:

    python -m sightseq.preprocess --task text_recognition \
        --trainpref [DATA]/train.txt \
        --destdir [DATA] --padding-factor 1

## Training

Training strategy (Attention):

    python -m sightseq.train [DATA] \
        --task text_recognition --arch decoder_attention \
        --decoder-layers 2 --batch-size 16 --dropout 0.0 \
        --max-epoch 100 --criterion cross_entropy --num-workers 4 \
        --optimizer adam --adam-eps 1e-04 --lr 0.001 --min-lr 1e-09 \
        --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0 \
        --no-token-crf --save-interval 1

Training strategy (Transformer):

    python -m sightseq.train [DATA] \
        --task text_recognition --arch decoder_transformer \
        --batch-size 16 --dropout 0.0  --max-epoch 100 \
        --criterion cross_entropy \
        --num-workers 4 --optimizer adam --decoder-layers 2 \
        --adam-eps 1e-04 --lr 0.001 --min-lr 1e-09 \
        --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --weight-decay 0.0 --no-token-crf --no-token-rnn \
        --save-interval 1 --encoder-normalize-before

Training strategy (CRNN):

    python -m sightseq.train [DATA] \
        --task text_recognition --arch decoder_crnn \
        --decoder-layers 2 --batch-size 16 \
        --max-epoch 50 --criterion ctc_loss --num-workers 4 \
        --optimizer adam --adam-eps 1e-04 --lr 0.001 --min-lr 1e-09 \
        --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0 \
        --save-interval 1

## Testing

Use trained model to test (Attention):

    python -m sightseq.generate [DATA] \
        --arch decoder_attention --path [CHECKPOINTS_DIR] \
        --task text_recognition \
        --buffer-size 16 --num-workers 4 --gen-subset valid \
        --beam 5 --batch-size 16 --quiet

Use trained model to test (Transformer):

    python -m sightseq.generate [DATA] \
        --arch decoder_transformer --path [CHECKPOINTS_DIR] \
        --task text_recognition \
        --buffer-size 16 --num-workers 4 --gen-subset valid \
        --batch-size 16 --beam 5 --quiet

Use trained model to test (CRNN):

    python -m sightseq.generate [DATA] \
        --arch decoder_crnn --path [CHECKPOINTS_DIR] \
        --task text_recognition --criterion ctc_loss \
        --sacrebleu \
        --buffer-size 16 --num-workers 4 --gen-subset valid \
        --batch-size 16 --quiet
