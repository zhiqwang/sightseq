# Text Recognition

Example to train a text recognition model as described in [Shi et al. (2015), CRNN: An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717).

## Usage
- Navigate (`cd`) to the root of the toolbox `[YOUR_SIGHTSEQ_ROOT]`.
- Prepare Dataset

It is recommended to symlink the dataset root to `[YOUR_SIGHTSEQ_ROOT]/data-bin`.

```
.
├── data-bin
│   └── [DATA]
│       ├── images
│       ├── train.txt
│       ├── valid.txt
│       ├── test.txt
│       └── dict.txt (will be generated in the [preprocess strategy](#preprocess-strategy))
├── examples
└── sightseq
```

### Annotation file format

In each line in the annotation file, the format is:

```
img_path char1 char2 char3 char4 char5 ...
```

where the `char` is the sequence's character.

For example, there is an image named "00120_00091.jpg" in folder `[DATA]/images`, its constant is "hello world", there should be a line in the `[DATA]/train.txt` or `[DATA]/valid.txt`.

```
00120_00091.jpg h e l l o w o r l d
```

## Preprocess Strategy
Generate `dict.txt`:

```
python -m sightseq.preprocess --task text_recognition \
    --trainpref [DATA]/train.txt \
    --destdir [DATA] --padding-factor 1
```

## Training
Training strategy (Attention):

```
python -m sightseq.train [DATA] \
    --task text_recognition --arch decoder_attention \
    --decoder-layers 2 --batch-size 16 --dropout 0.0 \
    --max-epoch 100 --criterion cross_entropy --num-workers 4 \
    --optimizer adam --adam-eps 1e-04 --lr 0.001 --min-lr 1e-09 \
    --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0 \
    --no-token-crf --save-interval 1
```

Training strategy (Transformer):

```
python -m sightseq.train [DATA] \
    --task text_recognition --arch decoder_transformer \
    --batch-size 16 --dropout 0.0  --max-epoch 100 \
    --criterion cross_entropy \
    --num-workers 4 --optimizer adam --decoder-layers 2 \
    --adam-eps 1e-04 --lr 0.001 --min-lr 1e-09 \
    --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --weight-decay 0.0 --no-token-crf --no-token-rnn \
    --save-interval 1 --encoder-normalize-before
```

Training strategy (CRNN):

```
python -m sightseq.train [DATA] \
    --task text_recognition --arch decoder_crnn \
    --decoder-layers 2 --batch-size 16 \
    --max-epoch 50 --criterion ctc_loss --num-workers 4 \
    --optimizer adam --adam-eps 1e-04 --lr 0.001 --min-lr 1e-09 \
    --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0 \
    --save-interval 1
```

## Testing
Use trained model to test (Attention):

```
python -m sightseq.generate_ocr [DATA] \
    --arch decoder_attention --path [CHECKPOINTS_DIR] \
    --task text_recognition \
    --buffer-size 16 --num-workers 4 --gen-subset valid \
    --beam 5 --batch-size 16 --quiet
```

Use trained model to test (Transformer):

```
python -m sightseq.generate_ocr [DATA] \
    --arch decoder_transformer --path [CHECKPOINTS_DIR] \
    --task text_recognition \
    --buffer-size 16 --num-workers 4 --gen-subset valid \
    --batch-size 16 --beam 5 --quiet
```

Use trained model to test (CRNN):

```
python -m sightseq.generate_ocr [DATA] \
    --arch decoder_crnn --path [CHECKPOINTS_DIR] \
    --task text_recognition --criterion ctc_loss \
    --sacrebleu \
    --buffer-size 16 --num-workers 4 --gen-subset valid \
    --batch-size 16 --quiet
```
