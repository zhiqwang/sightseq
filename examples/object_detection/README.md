# Object Detection

Example to train a object detection model as described in [Ren et al. (2015), Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497).

## Detection Performance
I port the torchvision's fasterrcnn_resnet50_fpn weights to sightseq. The result will be reported here soon.

## Installation Requirements

- python >= 3.5
- pytorch >= 1.1.0
- torchvision >= 0.3.0
- cocoapi

cocoapi step-by-step installation
```shell
# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
```

## Usage
- Navigate (`cd`) to the root of the toolbox `[YOUR_SIGHTSEQ_ROOT]`.

## Prepare COCO Dataset
It is recommended to symlink the dataset root to `$SIGHTSEQ/data-bin`.

```
sightseq
├── sightseq
├── examples
├── data-bin
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

## Training
```
python -m sightseq.train [DATA] \
    --task object_detection \
    --arch fasterrcnn_resnet50_fpn \
    --criterion fasterrcnn_loss \
    --optimizer sgd --lr 0.02 \
    --batch-size 2 --valid-subset val \
    --max-epoch 24 --no-progress-bar
```

## Testing
```
python -m sightseq.generate_coco [DATA] \
    --task object_detection \
    --batch-size 2 --gen-subset val \
    --path checkpoints/checkpoint_last.pt 
```