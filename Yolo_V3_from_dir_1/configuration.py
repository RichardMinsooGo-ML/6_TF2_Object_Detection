# dataset_name = "1K_MNIST"
# dataset_name = "1K_Fashion_MNIST"
# dataset_name = "1K_VOC"
# dataset_name = "1K_COCO"

# dataset_name = "14K_MNIST"
# dataset_name = "14K_Fashion_MNIST"
# dataset_name = "14K_VOC"
# dataset_name = "14K_COCO"

# dataset_name = "117K_MNIST"
# dataset_name = "117K_Fashion_MNIST"
# dataset_name = "117K_COCO"
# dataset_name = "100K_OID_v6"

TRAIN_FROM_CHECKPOINT       = True # "saved_model/yolov3_custom"
SIZE_TRAIN = 1024*16
SIZE_TEST  = 512*4

YOLO_CUSTOM_WEIGHTS         = False # "checkpoints/yolov3_custom" # used in evaluate_mAP.py and custom model detection, if not using leave False
                            # YOLO_CUSTOM_WEIGHTS also used with TensorRT and custom model detection    
# YOLO options
YOLO_TYPE                   = "yolov3" # yolov4 or yolov3
YOLO_FRAMEWORK              = "tf" # "tf" or "trt"
YOLO_V3_WEIGHTS             = "../00_Darknet_Weights/yolov3.weights"
YOLO_V3_TINY_WEIGHTS        = "../00_Darknet_Weights/yolov3-tiny.weights"
# YOLO_TRT_QUANTIZE_MODE      = "INT8" # INT8, FP16, FP32
YOLO_COCO_CLASSES           = "../04_1K_COCO2017/coco.names"

if dataset_name == "1K_MNIST":
    TRAIN_CLASSES               = "../01_1K_MNIST/mnist.names"
    TRAIN_ANNOT_PATH            = "../01_1K_MNIST/mnist_train.txt"
    TEST_ANNOT_PATH             = "../01_1K_MNIST/mnist_val.txt"
    TRAIN_CHECKPOINTS_FOLDER    = "../checkpoints/V3_weight_MNIST"
    DATA_TYPE = "yolo_v3_mnist"

elif dataset_name == "1K_Fashion_MNIST":
    TRAIN_CLASSES               = "../02_1K_Fashion_MNIST/mnist.names"
    TRAIN_ANNOT_PATH            = "../02_1K_Fashion_MNIST/mnist_train.txt"
    TEST_ANNOT_PATH             = "../02_1K_Fashion_MNIST/mnist_val.txt"
    TRAIN_CHECKPOINTS_FOLDER    = "../checkpoints/V3_weight_fashion_MNIST"
    DATA_TYPE = "yolo_v3_fashion_mnist"

elif dataset_name == "1K_VOC":
    TRAIN_CLASSES               = "../03_1K_VOC2012/voc2012.names"
    TRAIN_ANNOT_PATH            = "../03_1K_VOC2012/VOC2012_train.txt"
    TEST_ANNOT_PATH             = "../03_1K_VOC2012/VOC2012_val.txt"
    TRAIN_CHECKPOINTS_FOLDER    = "../checkpoints/V3_weight_VOC"
    DATA_TYPE = "yolo_v3_voc"
    
elif dataset_name == "1K_COCO":
    TRAIN_CLASSES               = "../04_1K_COCO2017/coco.names"
    TRAIN_ANNOT_PATH            = "../04_1K_COCO2017/COCO2017_train.txt"
    TEST_ANNOT_PATH             = "../04_1K_COCO2017/COCO2017_val.txt"
    TRAIN_CHECKPOINTS_FOLDER    = "../checkpoints/V3_weight_COCO"
    DATA_TYPE = "yolo_v3_coco"
    
elif dataset_name == "14K_MNIST":
    TRAIN_CLASSES               = "../11_14K_MNIST/mnist.names"
    TRAIN_ANNOT_PATH            = "../11_14K_MNIST/mnist_train.txt"
    TEST_ANNOT_PATH             = "../11_14K_MNIST/mnist_val.txt"
    TRAIN_CHECKPOINTS_FOLDER    = "../checkpoints/V3_weight_MNIST"
    DATA_TYPE = "yolo_v3_mnist"

elif dataset_name == "14K_Fashion_MNIST":
    TRAIN_CLASSES               = "../12_14K_Fashion_MNIST/mnist.names"
    TRAIN_ANNOT_PATH            = "../12_14K_Fashion_MNIST/mnist_train.txt"
    TEST_ANNOT_PATH             = "../12_14K_Fashion_MNIST/mnist_val.txt"
    TRAIN_CHECKPOINTS_FOLDER    = "../checkpoints/V3_weight_fashion_MNIST"
    DATA_TYPE = "yolo_v3_fashion_mnist"

elif dataset_name == "14K_VOC":
    TRAIN_CLASSES               = "../13_14K_VOC2012/voc2012.names"
    TRAIN_ANNOT_PATH            = "../13_14K_VOC2012/VOC2012_train.txt"
    TEST_ANNOT_PATH             = "../13_14K_VOC2012/VOC2012_val.txt"
    TRAIN_CHECKPOINTS_FOLDER    = "../checkpoints/V3_weight_VOC"
    DATA_TYPE = "yolo_v3_voc"
    
elif dataset_name == "14K_COCO":
    TRAIN_CLASSES               = "../14_14K_COCO2017/coco.names"
    TRAIN_ANNOT_PATH            = "../14_14K_COCO2017/COCO2017_train.txt"
    TEST_ANNOT_PATH             = "../14_14K_COCO2017/COCO2017_val.txt"
    TRAIN_CHECKPOINTS_FOLDER    = "../checkpoints/V3_weight_COCO"
    DATA_TYPE = "yolo_v3_coco"
    
elif dataset_name == "117K_MNIST":
    TRAIN_CLASSES               = "../21_117K_MNIST/mnist.names"
    TRAIN_ANNOT_PATH            = "../21_117K_MNIST/mnist_train.txt"
    TEST_ANNOT_PATH             = "../21_117K_MNIST/mnist_val.txt"
    TRAIN_CHECKPOINTS_FOLDER    = "../checkpoints/V3_weight_MNIST"
    DATA_TYPE = "yolo_v3_mnist"

elif dataset_name == "117K_Fashion_MNIST":
    TRAIN_CLASSES               = "../22_117K_Fashion_MNIST/mnist.names"
    TRAIN_ANNOT_PATH            = "../22_117K_Fashion_MNIST/mnist_train.txt"
    TEST_ANNOT_PATH             = "../22_117K_Fashion_MNIST/mnist_val.txt"
    TRAIN_CHECKPOINTS_FOLDER    = "../checkpoints/V3_weight_fashion_MNIST"
    DATA_TYPE = "yolo_v3_fashion_mnist"

elif dataset_name == "117K_COCO":
    TRAIN_CLASSES               = "../24_117K_COCO2017/coco.names"
    TRAIN_ANNOT_PATH            = "../24_117K_COCO2017/COCO2017_train.txt"
    TEST_ANNOT_PATH             = "../24_117K_COCO2017/COCO2017_val.txt"
    TRAIN_CHECKPOINTS_FOLDER    = "../checkpoints/V3_weight_COCO"
    DATA_TYPE = "yolo_v3_coco"
        
elif dataset_name == "100K_OID_v6":
    TRAIN_CLASSES               = "../36_100K_OID_v6/OID_V6.names"
    TRAIN_ANNOT_PATH            = "../36_100K_OID_v6/OID_V6_train.txt"
    TEST_ANNOT_PATH             = "../36_100K_OID_v6/OID_V6_test.txt"
    TRAIN_CHECKPOINTS_FOLDER    = "../checkpoints/V3_weight_OID_30_cls"
    DATA_TYPE = "yolo_v3_OID_30_cls"
    

YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 416

"""
if YOLO_TYPE                == "yolov4":
    YOLO_ANCHORS            = [[[12,  16], [19,   36], [40,   28]],
                               [[36,  75], [76,   55], [72,  146]],
                               [[142,110], [192, 243], [459, 401]]]
"""
if YOLO_TYPE                == "yolov3":
    YOLO_ANCHORS            = [[[10,  13], [16,   30], [33,   23]],
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]]
# Train options
TRAIN_YOLO_TINY             = False
TRAIN_SAVE_BEST_ONLY        = True # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT       = False # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_LOGDIR                = "log"
TRAIN_MODEL_NAME            = f"{YOLO_TYPE}"+"_"+dataset_name
TRAIN_LOAD_IMAGES_TO_RAM    = True # With True faster training, but need more RAM
TRAIN_BATCH_SIZE            = 4
TRAIN_INPUT_SIZE            = 416
TRAIN_DATA_AUG              = True
TRAIN_TRANSFER              = True
TRAIN_LR_INIT               = 1e-4
TRAIN_LR_END                = 1e-6
TRAIN_WARMUP_EPOCHS         = 1
TRAIN_EPOCHS                = 20

# TEST options
TEST_BATCH_SIZE             = 4
TEST_INPUT_SIZE             = 416
TEST_DATA_AUG               = False
TEST_DECTECTED_IMAGE_PATH   = ""
TEST_SCORE_THRESHOLD        = 0.3
TEST_IOU_THRESHOLD          = 0.45

"""
#YOLOv3-TINY and YOLOv4-TINY WORKAROUND
if TRAIN_YOLO_TINY:
    YOLO_STRIDES            = [16, 32, 64]    
    YOLO_ANCHORS            = [[[10,  14], [23,   27], [37,   58]],
                               [[81,  82], [135, 169], [344, 319]],
                               [[0,    0], [0,     0], [0,     0]]]
"""