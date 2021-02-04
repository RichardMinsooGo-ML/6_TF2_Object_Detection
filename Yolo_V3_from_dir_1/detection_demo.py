
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolo_core.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from configuration import *

image_path   = "./IMAGES/kite.jpg"
video_path   = "./IMAGES/test.mp4"

yolo = Load_Yolo_model()
# detect_image(yolo, image_path, "./IMAGES/kite_pred.jpg", input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,0))
detect_video(yolo, video_path, "./IMAGES/tracking_results_2.mp4", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255, 0, 0))

# detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,0), realtime=False)
