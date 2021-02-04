# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from configuration import *

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, Input, Lambda,
    LeakyReLU, UpSampling2D, 
    MaxPool2D, 
    concatenate, Add, ZeroPadding2D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)

# %matplotlib inline

print(tf.__version__)

# anchor boxes
YOLO_ANCHORS = np.array(
    [(10, 13), (16, 30), (33, 23), 
     (30, 61), (62, 45), (59, 119),
     (116, 90), (156, 198), (373, 326)], np.float32) / 416

YOLO_ANCHORS_MASKS = np.array([[6, 7, 8],
                               [3, 4, 5],
                               [0, 1, 2]])

num_max_box = 100

# Use deep copy()
anchors = YOLO_ANCHORS
anchor_masks = YOLO_ANCHORS_MASKS

# Build the Model
# Darknet53
class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Make trainable=False freeze BN for real (the og version is sad)
    """
    
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)
    
def DarknetConv2D(x, filters, size, stride=1, batch_norm=True):
    if stride == 1:
        padding = 'same'
    else:
        # downsample
        # padding=((top_pad, bottom_pad), (left_pad, right_pad))
        x = ZeroPadding2D(((1, 0), (1, 0)))(x) # top left half-padding
        padding = 'valid'
    
    x= Conv2D(filters=filters, kernel_size=size,
              strides=(stride, stride), padding=padding,
              use_bias=not batch_norm, 
              kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    
    return x

def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv2D(x, filters // 2, 1)
    x = DarknetConv2D(x, filters, 3)
    x = Add()([prev, x])
    return x

# ResidualBlock
def DarknetBlock(x, filters, num_blocks):
    x = DarknetConv2D(x, filters, 3, stride=2)
    for _ in range(num_blocks):
        x = DarknetResidual(x, filters)
    return x

def darknet_body(name=None):
    x = inputs = Input([None, None, 3])
    
    # Darknet53
    x = DarknetConv2D(x, 32, 3)
    x = DarknetBlock(x, 64, num_blocks=1)
    x = DarknetBlock(x, 128, num_blocks=2)
    x = x_36 = DarknetBlock(x, 256, num_blocks=8) # skip connection
    x = x_61 = DarknetBlock(x, 512, num_blocks=8) # conv + residual
    x = DarknetBlock(x, 1024, num_blocks=4) # x_74
    
    return Model(inputs, (x_36, x_61, x), name=name)

def yolo_body(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs
            
            # concat with skip connection
            x = DarknetConv2D(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = concatenate([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
        
        x = DarknetConv2D(x, filters, 1)
        x = DarknetConv2D(x, filters * 2, 3)
        x = DarknetConv2D(x, filters, 1)
        x = DarknetConv2D(x, filters * 2, 3)
        x = DarknetConv2D(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv

def yolo_output(filters, num_anchors, classes, name=None):
    def yolo_output_conv(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv2D(x, filters * 2, 3)
        x = DarknetConv2D(x, (num_anchors * (classes + 5)), 1, batch_norm=False)
        # output
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], 
                                            num_anchors, classes + 5)))(x)
        return Model(inputs, x, name=name)(x_in)
    return yolo_output_conv

def Yolov3(size=None, channels=3, 
           anchors=YOLO_ANCHORS, masks=YOLO_ANCHORS_MASKS,
           classes=80):    
    x = inputs = Input([size, size, channels], name='input')
    
    # Darknet53
    '''
    x = DarknetConv2D(x, 32, 3)
    x = DarknetBlock(x, 64, num_blocks=1)
    x = DarknetBlock(x, 128, num_blocks=2)
    x = x_36 = DarknetBlock(x, 256, num_blocks=8) # skip connection
    x = x_61 = DarknetBlock(x, 512, num_blocks=8) # conv + residual
    x = DarknetBlock(x, 1024, num_blocks=4) # x_74
    '''
    x_36, x_61, x = darknet_body(name='yolo_darknet')(x)
    
    ##############################################################################
    # Yolo Body
    '''
    x = DarknetConv2D(x, 512, 1)
    x = DarknetConv2D(x, 1024, 3)
    x = DarknetConv2D(x, 512, 1)
    x = DarknetConv2D(x, 1024, 3)
    x = x_79 = DarknetConv2D(x, 512, 1)
    
    # Yolo Output 1. 13x13x(anchor*(classes+5)
    x = DarknetConv2D(x, 1024, 3)
    x = DarknetConv2D(x, (num_anchors * (classes + 5)), 1, batch_norm=False)
    output_0 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], 
                                               num_anchors, classes + 5)))(x)
    '''
    x = yolo_body(512, name='yolo_conv_0')(x)
    output_0 = yolo_output(512, len(masks[0]), classes, name='yolo_output_0')(x)
                           
    ############################################################################## 
    '''
    # 82, output_0
    # 83, route -4 -> x_79
    # x_79 upsample + x_61
    x = DarknetConv2D(x_79, 256, 1) # x_84
    x = UpSampling2D(2)(x)
    x = concatenate([x, x_61]) 
    
    # Yolo Body
    x = DarknetConv2D(x, 256, 1)
    x = DarknetConv2D(x, 512, 3)
    x = DarknetConv2D(x, 256, 1)
    x = DarknetConv2D(x, 512, 3)
    x = x_91 = DarknetConv2D(x, 256, 1) 
    
    # Yolo Output 2. 26x26x(anchor*(classes+5)
    x = DarknetConv2D(x, 512, 3)
    x = DarknetConv2D(x, (num_anchors * (classes + 5)), 1, batch_norm=False)             
    output_1 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], 
                                               num_anchors, classes + 5)))(x)
    '''
    x = yolo_body(256, name='yolo_conv_1')((x, x_61))
    output_1 = yolo_output(256, len(masks[1]), classes, name='yolo_output_1')(x)
    
    ##############################################################################  
    '''
    # 94, output_1
    # 95. route -4 -> x_91
    # x_91 upsample + x_36
    x = DarknetConv2D(x_91, 128, 1) # x_92
    x = UpSampling2D(2)(x)
    x = concatenate([x, x_36])
    
    # Yolo Body
    x = DarknetConv2D(x, 128, 1)
    x = DarknetConv2D(x, 256, 3)
    x = DarknetConv2D(x, 128, 1)
    x = DarknetConv2D(x, 256, 3)
    x = x_91 = DarknetConv2D(x, 128, 1)
    
    # Yolo Output 3. 52x52x(anchor*(classes+5)
    x = DarknetConv2D(x, 256, 3)
    x = DarknetConv2D(x, (num_anchors * (classes + 5)), 1, batch_norm=False)
    output_2 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], 
                                               num_anchors, classes + 5)))(x)
    '''
    x = yolo_body(128, name='yolo_conv_2')((x, x_36)) # x_103
    # Yolo Output 3. 52x52x(anchor*(classes+5)
    output_2 = yolo_output(128, len(masks[2]), classes, name='yolo_output_2')(x) # x_106

    return Model(inputs, (output_0, output_1, output_2), name='yolov3')

model = Yolov3(416, classes = n_classes)
model.summary()

# YOLOv3 LOSS
def yolo_bboxes(pred, anchors, classes):
    """YOLO bounding box formula

    bx = sigmoid(tx) + cx
    by = sigmoid(ty) + cy
    bw = pw * exp^(tw)
    bh = ph * exp^(th)
    Pr(obj) * IOU(b, object) = sigmoid(to) # confidence

    (tx, ty, tw, th, to) are the output of the model.
    """
    # pred: (batch_size, grid, grid, anchors, (tx, ty, tw, th, conf, ...classes))
    grid_size = tf.shape(pred)[1]

    box_xy = tf.sigmoid(pred[..., 0:2])
    box_wh = pred[..., 2:4]
    box_confidence = tf.sigmoid(pred[..., 4:5])
    box_class_probs = tf.sigmoid(pred[..., 5:])
    # Darknet raw box
    pred_raw_box = tf.concat((box_xy, box_wh), axis=-1)

    # box_xy: (grid_size, grid_size, num_anchors, 2)
    # grid: (grdid_siez, grid_size, 1, 2)
    #       -> [0,0],[0,1],...,[0,12],[1,0],[1,1],...,[12,12]
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors
    pred_box = tf.concat((box_xy, box_wh), axis=-1)

    return pred_box, box_confidence, box_class_probs, pred_raw_box
    
yolo = Yolov3(classes=n_classes)
yolo.load_weights(model_ckpt)

def yolo_boxes_and_scores(yolo_output, anchors, classes=n_classes):
    # yolo_boxes: pred_box, box_confidence, box_class_probs, pred_raw_box
    pred_box, box_confidence, box_class_probs, pred_raw_box = yolo_bboxes(yolo_output, anchors, classes)

    # Convert boxes to be ready for filtering functions.
    # Convert YOLO box predicitions to bounding box corners.
    # (x, y, w, h) -> (x1, y1, x2, y2)
    box_xy = pred_box[..., 0:2]
    box_wh = pred_box[..., 2:4]
    box_x1y1 = box_xy - (box_wh / 2.)
    box_x2y2 = box_xy + (box_wh / 2.)
    boxes = tf.concat([box_x1y1, box_x2y2], axis=-1)
    boxes = tf.reshape(boxes, [-1, 4])

    # Compute box scores
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, classes])
    return boxes, box_scores

def yolo_non_max_suppression(boxes, box_scores, 
                             classes=n_classes, 
                             max_boxes=100,
                             score_threshold=0.5,
                             iou_threshold=0.5):
    """Perform Score-filtering and Non-max suppression

    boxes: (10647, 4)
    box_scores: (10647, 80)
    # 10647 = (13*13 + 26*26 + 52*52) * 3(anchor)
    """

    # Create a mask, same dimension as box_scores.
    mask = box_scores >= score_threshold # (10647, 80)

    output_boxes = []
    output_scores = []
    output_classes = []

    # Perform NMS for all classes
    for c in range(classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        selected_indices = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes, iou_threshold)

        class_boxes = tf.gather(class_boxes, selected_indices)
        class_box_scores = tf.gather(class_box_scores, selected_indices)

        classes = tf.ones_like(class_box_scores, 'int32') * c

        output_boxes.append(class_boxes)
        output_scores.append(class_box_scores)
        output_classes.append(classes)

    output_boxes = tf.concat(output_boxes, axis=0)
    output_scores = tf.concat(output_scores, axis=0)
    output_classes = tf.concat(output_classes, axis=0)

    return output_scores, output_boxes, output_classes

def yolo_eval(yolo_outputs, 
              image_shape=(416, 416), 
              classes=n_classes, 
              max_boxes=100, 
              score_threshold=0.5, 
              iou_threshold=0.5):
    # Retrieve outputs of the YOLO model.
    for i in range(0,3):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[i], anchors[6-3*i:9-3*i], classes)
        if i == 0:
            boxes, box_scores = _boxes, _box_scores
        else:
            boxes = tf.concat([boxes, _boxes], axis=0)
            box_scores = tf.concat([box_scores, _box_scores], axis=0)

    # Perform Score-filtering and Non-max suppression
    scores, boxes, classes = yolo_non_max_suppression(boxes, box_scores,
                                                      classes,
                                                      max_boxes,
                                                      score_threshold,
                                                      iou_threshold)

    return scores, boxes, classes

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height, width = image_shape
    image_dims = tf.stack([width, height, width, height])
    image_dims = tf.cast(tf.reshape(image_dims, [1, 4]), tf.float32)
    boxes = boxes * image_dims
    return boxes

import colorsys
import random

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def draw_outputs(image, outputs, class_names, colors):
    h, w, _ = image.shape
    scores, boxes, classes = outputs
    boxes = scale_boxes(boxes, (h, w))

    for i in range(scores.shape[0]):
        left, top, right, bottom = boxes[i]
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
        right = min(w, np.floor(right + 0.5).astype('int32'))
        class_id = int(classes[i])
        predicted_class = class_names[class_id]
        score = scores[i].numpy()

        label = '{} {:.2f}'.format(predicted_class, score)

        # colors: RGB
        cv2.rectangle(image, (left, top), (right, bottom), tuple(colors[class_id]), 6)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2

        label_size = cv2.getTextSize(label, font_face, font_scale, font_thickness)[0]
        label_rect_left, label_rect_top = int(left - 3), int(top - 3)
        label_rect_right, label_rect_bottom = int(left + 3 + label_size[0]), int(top - 5 - label_size[1])
        cv2.rectangle(image, (label_rect_left, label_rect_top), (label_rect_right, label_rect_bottom),
                      tuple(colors[class_id]), -1)

        cv2.putText(image, label, (left, int(top - 4)),
                    font_face, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        
    return image

import time

# Pick an image
video_path = "./data/test.mp4"
output_path = "./data/test_output.avi"
show=False

# -------------------------------
times, times_2 = [], []
vid = cv2.VideoCapture(video_path)

# by default VideoCapture returns float instead of int
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'XVID')
# codec = cv2.VideoWriter_fourcc(*'MP4V')   # try with *.mp4 for output_path
# codec = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

while True:
    _, img = vid.read()

    try:
        images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    except:
        break

    # image_data = image_preprocess(np.copy(images), [input_size, input_size])
    # image_data = image_data[np.newaxis, ...].astype(np.float32)
    
    # Pre-processing of image input
    image_data = cv2.resize(images, (416, 416))   # Modify the input image size to meet the requirements of the model
    image_data = image_data / 255.               # Perform image normalization
    image_data = np.expand_dims(image_data, 0)   # increase batch dimension


    t1 = time.time()
    
    # -------------------------------

    # Perform image detection
    yolo_outputs = yolo.predict(image_data)
    
    t2 = time.time()

    scores, boxes, classes = yolo_eval( yolo_outputs, score_threshold=0.2, iou_threshold=0.2)

    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    
    """
    print("detections:")
    for i in range(scores.shape[0]):
        print("\t{}, {}, {}".format(
            class_names[int(classes[i])], scores[i], boxes[i]
        ))
    
    # Save
    cv2.imwrite("./data/output_dog.jpg", image)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    """
    # Draw bounding boxes on the image file
    image = draw_outputs(images, (scores, boxes, classes), class_names, colors)

    t3 = time.time()
    times.append(t2-t1)
    times_2.append(t3-t1)

    times = times[-20:]
    times_2 = times_2[-20:]

    ms = sum(times)/len(times)*1000
    fps = 1000 / ms
    fps2 = 1000 / (sum(times_2)/len(times_2)*1000)

    image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))

    print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
    if output_path != '': out.write(image)
    if show:
        cv2.imshow('output', image)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

cv2.destroyAllWindows()









