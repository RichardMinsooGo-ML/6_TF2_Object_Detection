#================================================================
#
#   File name   : detect_mnist.py
#   Author      : PyLessons
#   Created date: 2020-08-12
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : mnist object detection example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import random
import time
import tensorflow as tf
from yolo_core.yolov4 import Create_Yolo
from yolo_core.utils import detect_image
from configuration import *

n_test = 10

for idx in range(n_test):
    ID = random.randint(0, 200)
    label_txt = "../01_1K_MNIST/mnist_val.txt"
    image_info = open(label_txt).readlines()[ID].split()

    image_path = image_info[0]

    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    # yolo.load_weights(f"./saved_model/{TRAIN_MODEL_NAME}") # use keras weights
    
    save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, DATA_TYPE)
    yolo.load_weights(save_directory)

    detect_image(yolo, image_path, "./IMAGES/output_mnist"+str(idx).zfill(2)+".jpg", input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
    print(idx+1,"/",n_test,"Converted")

    