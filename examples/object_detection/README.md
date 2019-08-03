# Object Detection

Example to train a object detection model as described in [Ren et al. (2015), Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497).

## Detection Performance
The architecture of `sightseq` object detection is equal to `torchvision` apart from spliting the computation of criterion from the network's forward propagation graph to [`fasterrcnn_loss`](../../sightseq/criterions/fasterrcnn_loss.py). So the models is backwards-compatible with `torchvison`. And I borrow the `torchvision` fasterrcnn_resnet50_fpn weights to `sightseq`, the mAP is exactly the same as `torchvision`.

Run `eval`, you can get the mAP on the `coco_2017_val`.
```
python -m examples.object_detection.eval [COCO_DATA_PATH] \
    --task object_detection --num-classes 91 \
    --arch fasterrcnn_resnet50_fpn --criterion fasterrcnn_loss \
    --optimizer sgd --lr 0.02 --momentum 0.9 --weight-decay 1e-4 \
    --batch-size 1 --valid-subset val --pretrained
```

Model | box AP
--- | ---
fasterrcnn_resnet50_fpn | 0.37

## Installation Requirements

- python >= 3.5
- pytorch >= 1.1.0
- torchvision >= 0.3.0
- cocoapi

cocoapi step-by-step installation:
```shell
# install pycocotools
# cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
```

## Prepare COCO Dataset
It is recommended to symlink the dataset root to `$[SIGHTSEQ]/data-bin`.

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
*Note*, currently the data sampler with aspect ratio group is not ensembled to `sightseq` when training, but I keep this parameter for later use.
```
python -m sightseq.train [DATA] \
    --task object_detection \
    --num-classes 91 \
    --arch fasterrcnn_resnet50_fpn \
    --criterion fasterrcnn_loss \
    --optimizer sgd \
    --lr 0.02 --momentum 0.9 --weight-decay 1e-4 \
    --batch-size 2 \
    --valid-subset val \
    --max-epoch 26 \
    --aspect-ratio-group-factor 3 \
    --no-progress-bar
```

## Testing
```
python -m sightseq.generate_coco [DATA] \
    --task object_detection \
    --criterion fasterrcnn_loss \
    --batch-size 1 --gen-subset val \
    --path [CHECKPOINTS_PATH]
```
