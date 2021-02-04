#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict

dataset_name = "MNIST"
# dataset_name = "VOC"
# dataset_name = "COCO"

__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()
__C.TRAIN                     = edict()
__C.TEST                      = edict()

# Set the class name
if dataset_name == "MNIST":
    __C.YOLO.CLASSES              = "../01_1K_MNIST/mnist.names"
    __C.TRAIN.ANNOT_PATH          = "../01_1K_MNIST/mnist_train.txt"
    __C.TEST.ANNOT_PATH           = "../01_1K_MNIST/mnist_val.txt"

elif dataset_name == "VOC":
    __C.YOLO.CLASSES              = "../03_mini_VOC2012/voc2012.names"
    __C.TRAIN.ANNOT_PATH          = "../03_mini_VOC2012/VOC2012_train.txt"
    __C.TEST.ANNOT_PATH           = "../03_mini_VOC2012/VOC2012_val.txt"

elif dataset_name == "COCO":
    __C.YOLO.CLASSES              = "../04_mini_COCO2017/coco.names"
    __C.TRAIN.ANNOT_PATH          = "../04_mini_COCO2017/COCO2017_train.txt"
    __C.TEST.ANNOT_PATH           = "../04_mini_COCO2017/COCO2017_val.txt"
    
__C.YOLO.ANCHORS              = [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
__C.YOLO.ANCHORS_V3           = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
__C.YOLO.ANCHORS_TINY         = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.STRIDES_TINY         = [16, 32]
__C.YOLO.XYSCALE              = [1.2, 1.1, 1.05]
__C.YOLO.XYSCALE_TINY         = [1.05, 1.05]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5

# Train options
__C.TRAIN.BATCH_SIZE          = 2
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = 416
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 2
__C.TRAIN.SECOND_STAGE_EPOCHS   = 20


# TEST options
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 416
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.25
__C.TEST.IOU_THRESHOLD        = 0.5


