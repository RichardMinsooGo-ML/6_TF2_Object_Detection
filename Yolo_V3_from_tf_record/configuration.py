import tensorflow as tf
import os

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

# dataset_name = "64K_MNIST"
# dataset_name = "64K_Fashion_MNIST"
# dataset_name = "48K_COCO"

dataset_name = "Demo"

if dataset_name == "Demo":
    data_dir = '../04_1K_COCO2017/'
    classes  = '../04_1K_COCO2017/coco.names'
    
    model_ckpt_dir = '../checkpoints/darknet_tf_v3'
    type_dir = 'type_VOC_COCO'
    
if dataset_name == "1K_MNIST":
    data_dir = '../01_1K_MNIST/'
    classes  = '../01_1K_MNIST/MNIST.names'

    model_ckpt_dir = '../checkpoints/v3_tf_weight_MNIST'
    type_dir = 'type_MNIST'

if dataset_name == "1K_Fashion_MNIST":
    data_dir = '../02_1K_Fashion_MNIST/'
    classes  = '../02_1K_Fashion_MNIST/MNIST.names'
    
    model_ckpt_dir = '../checkpoints/v3_tf_weight_fashion_MNIST'
    type_dir = 'type_MNIST'
    
elif dataset_name == "1K_VOC":
    data_dir = '../03_1K_VOC2012/'
    classes  = '../03_1K_VOC2012/voc2012.names'
    
    model_ckpt_dir = '../checkpoints/v3_tf_weight_VOC_2012'
    type_dir = 'type_VOC_COCO'
    
elif dataset_name == "1K_COCO":
    data_dir = '../04_1K_COCO2017/'
    classes  = '../04_1K_COCO2017/coco.names'

    model_ckpt_dir = '../checkpoints/v3_tf_weight_COCO_2017'
    type_dir = 'type_VOC_COCO'

elif dataset_name == "14K_MNIST":
    data_dir = '../11_14K_MNIST/'
    classes  = '../11_14K_MNIST/MNIST.names'

    model_ckpt_dir = '../checkpoints/v3_tf_weight_MNIST'
    type_dir = 'type_MNIST'

elif dataset_name == "14K_Fashion_MNIST":
    data_dir = '../12_14K_Fashion_MNIST/'
    classes  = '../12_14K_Fashion_MNIST/MNIST.names'
    
    model_ckpt_dir = '../checkpoints/v3_tf_weight_fashion_MNIST'
    type_dir = 'type_MNIST'
    
elif dataset_name == "14K_VOC":
    data_dir = '../13_14K_VOC2012/'
    classes  = '../13_14K_VOC2012/voc2012.names'
    
    model_ckpt_dir = '../checkpoints/v3_tf_weight_VOC_2012'
    type_dir = 'type_VOC_COCO'
    
elif dataset_name == "14K_COCO":
    data_dir = '../14_14K_COCO2017/'
    classes  = '../14_14K_COCO2017/coco.names'

    model_ckpt_dir = '../checkpoints/v3_tf_weight_COCO_2017'
    type_dir = 'type_VOC_COCO'

elif dataset_name == "117K_MNIST":
    data_dir = '../21_117K_MNIST/'
    classes  = '../21_117K_MNIST/MNIST.names'

    model_ckpt_dir = '../checkpoints/v3_tf_weight_MNIST'
    type_dir = 'type_MNIST'

elif dataset_name == "117K_Fashion_MNIST":
    data_dir = '../22_117K_Fashion_MNIST/'
    classes  = '../22_117K_Fashion_MNIST/MNIST.names'
    
    model_ckpt_dir = '../checkpoints/v3_tf_weight_fashion_MNIST'
    type_dir = 'type_MNIST'
        
elif dataset_name == "117K_COCO":
    data_dir = '../24_117K_COCO2017/'
    classes  = '../24_117K_COCO2017/coco.names'

    model_ckpt_dir = '../checkpoints/v3_tf_weight_COCO_2017'
    type_dir = 'type_VOC_COCO'
    
elif dataset_name == "64K_MNIST":
    data_dir = '../31_64K_MNIST/'
    classes  = '../31_64K_MNIST/MNIST.names'

    model_ckpt_dir = '../checkpoints/v3_tf_weight_MNIST'
    type_dir = 'type_MNIST'

elif dataset_name == "64K_Fashion_MNIST":
    data_dir = '../32_64K_Fashion_MNIST/'
    classes  = '../32_64K_Fashion_MNIST/MNIST.names'
    
    model_ckpt_dir = '../checkpoints/v3_tf_weight_fashion_MNIST'
    type_dir = 'type_MNIST'
        
elif dataset_name == "48K_COCO":
    data_dir = '../34_48K_COCO2017/'
    classes  = '../34_48K_COCO2017/coco.names'

    model_ckpt_dir = '../checkpoints/v3_tf_weight_COCO_2017'
    type_dir = 'type_VOC_COCO'
        
if type_dir == 'type_MNIST':    
    image_dir_train = os.path.join(data_dir, 'mnist_train')
    annot_dir_train = os.path.join(data_dir, 'xml_train')

    image_dir_val = os.path.join(data_dir, 'mnist_val')
    annot_dir_val = os.path.join(data_dir, 'xml_val')
    
elif type_dir == 'type_VOC_COCO':
    image_dir_train = os.path.join(data_dir, 'train_images')
    annot_dir_train = os.path.join(data_dir, 'train_xml')

    image_dir_val = os.path.join(data_dir, 'val_images')
    annot_dir_val = os.path.join(data_dir, 'val_xml')

    
if dataset_name == "Demo":
    model_ckpt = os.path.join(model_ckpt_dir,'yolov3.tf')
else:
    model_ckpt = os.path.join(model_ckpt_dir,'yolov3_tf.tf')

tf_record_train = os.path.join(data_dir, 'Yolo_train.tfrecord')
tf_record_val   = os.path.join(data_dir, 'Yolo_val.tfrecord')

# load data from local
raw_train_dataset = tf.data.TFRecordDataset(tf_record_train)
raw_val_dataset   = tf.data.TFRecordDataset(tf_record_val)
class_names = [c.strip() for c in open(classes).readlines()]

# Common for all datasets
n_classes = len(class_names)
# print(len(class_names))
