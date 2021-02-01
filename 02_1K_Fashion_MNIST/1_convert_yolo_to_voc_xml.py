import os
import cv2
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
import numpy as np
from os.path import join
import shutil
## coco classes
YOLO_CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boot')

## converts the normalized positions  into integer positions
def unconvert(class_id, width, height, x, y, w, h):
    xmin = round(x*416)
    ymin = round(y*416)
    xmax = round(w*416)
    ymax = round(h*416)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)

## converts coco into xml 
def xml_transform(classes):  
    class_path  = join('txt_train')
    ids = list()
    l=os.listdir(class_path)
    
    check = '.DS_Store' in l
    if check == True:
        l.remove('.DS_Store')
        
    ids=[x.split('.')[0] for x in l]   

    annopath = join('txt_train', '%s.txt')
    imgpath = join('mnist_train', '%s.jpg')
    
    os.makedirs(join('xml_train'), exist_ok=True)
    outpath = join('xml_train', '%s.xml')

    for i in range(len(ids)):
        img_id = ids[i] 
        img= cv2.imread(imgpath % img_id)
        height, width, channels = img.shape # pega tamanhos e canais das images

        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'MNIST'
        img_name = img_id + '.jpg'
    
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = img_name
        
        node_source= SubElement(node_root, 'source')
        node_database = SubElement(node_source, 'database')
        node_database.text = 'MNIST database'
        
        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(width)
    
        node_height = SubElement(node_size, 'height')
        node_height.text = str(height)

        node_depth = SubElement(node_size, 'depth')
        node_depth.text = str(channels)

        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = '0'

        target = (annopath % img_id)
        if os.path.exists(target):
            label_norm= np.loadtxt(target).reshape(-1, 5)

            for i in range(len(label_norm)):
                labels_conv = label_norm[i]
                new_label = unconvert(labels_conv[0], width, height, labels_conv[1], labels_conv[2], labels_conv[3], labels_conv[4])
                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = classes[new_label[0]]
                
                node_pose = SubElement(node_object, 'pose')
                node_pose.text = 'Unspecified'
                
                
                node_truncated = SubElement(node_object, 'truncated')
                node_truncated.text = '0'
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'
                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(int(new_label[1]))
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(int(new_label[3]))
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text =  str(int(new_label[2]))
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(int(new_label[4]))
                xml = tostring(node_root, pretty_print=True)  
                dom = parseString(xml)
        
        if (i+1)%50 == 0:
            print("count :",i+1,"\n", xml)  
        f =  open(outpath % img_id, "wb")
        #f = open(os.path.join(outpath, img_id), "w")
        #os.remove(target)
        f.write(xml)
        f.close()     
        
xml_transform(YOLO_CLASSES)


## converts coco into xml 
def xml_transform_val(classes):  
    class_path  = join('txt_val')
    ids = list()
    l=os.listdir(class_path)
    
    check = '.DS_Store' in l
    if check == True:
        l.remove('.DS_Store')
        
    ids=[x.split('.')[0] for x in l]   

    annopath = join('txt_val', '%s.txt')
    imgpath = join('mnist_val', '%s.jpg')
    
    os.makedirs(join('xml_val'), exist_ok=True)
    outpath = join('xml_val', '%s.xml')

    for i in range(len(ids)):
        img_id = ids[i] 
        img= cv2.imread(imgpath % img_id)
        height, width, channels = img.shape # pega tamanhos e canais das images

        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'MNIST'
        img_name = img_id + '.jpg'
    
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = img_name
        
        node_source= SubElement(node_root, 'source')
        node_database = SubElement(node_source, 'database')
        node_database.text = 'MNIST database'
        
        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(width)
    
        node_height = SubElement(node_size, 'height')
        node_height.text = str(height)

        node_depth = SubElement(node_size, 'depth')
        node_depth.text = str(channels)

        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = '0'

        target = (annopath % img_id)
        if os.path.exists(target):
            label_norm= np.loadtxt(target).reshape(-1, 5)

            for i in range(len(label_norm)):
                labels_conv = label_norm[i]
                new_label = unconvert(labels_conv[0], width, height, labels_conv[1], labels_conv[2], labels_conv[3], labels_conv[4])
                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = classes[new_label[0]]
                
                node_pose = SubElement(node_object, 'pose')
                node_pose.text = 'Unspecified'
                
                
                node_truncated = SubElement(node_object, 'truncated')
                node_truncated.text = '0'
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'
                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(int(new_label[1]))
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(int(new_label[3]))
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text =  str(int(new_label[2]))
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(int(new_label[4]))
                xml = tostring(node_root, pretty_print=True)  
                dom = parseString(xml)
                
        if (i+1)%50 == 0:
            print("count :",i+1,"\n", xml)  
        f =  open(outpath % img_id, "wb")
        #f = open(os.path.join(outpath, img_id), "w")
        #os.remove(target)
        f.write(xml)
        f.close()     
        
xml_transform_val(YOLO_CLASSES)


# Delete the unused directories
# dirpath = os.path.join('dataset3', 'dataset')
dirpath = "txt_train"
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

dirpath = "txt_val"
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

if os.path.exists("tmp_mnist_train.txt"):
    os.remove("tmp_mnist_train.txt")

if os.path.exists("tmp_mnist_val.txt"):
    os.remove("tmp_mnist_val.txt")

    