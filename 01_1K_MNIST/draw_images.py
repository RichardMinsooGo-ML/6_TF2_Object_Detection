import os
import sys
import matplotlib.pyplot as plt
import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

dataset_path = './'
IMAGE_FOLDER = "mnist_train"
ANNOTATIONS_FOLDER = "xml_train"
n_plot = 16

FONT_FAMILY = "arial.ttf"
FONT_SIZE = 36
font = ImageFont.truetype(FONT_FAMILY, FONT_SIZE)
# font = ImageFont.truetype("arial.ttf", fontsize)


ann_root, ann_dir, ann_files = next(os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))
img_root, amg_dir, img_files = next(os.walk(os.path.join(dataset_path, IMAGE_FOLDER)))

count = 0

for xml_file in ann_files:
    
    count += 1

    # XML파일와 이미지파일은 이름이 같으므로, 확장자만 맞춰서 찾습니다.
    img_name = img_files[img_files.index(".".join([xml_file.split(".")[0], "jpg"]))]
    
    
    # print(os.path.splitext(img_name)[0])
    
    # import sys
    # sys.exit()
    
    
    img_file = os.path.join(img_root, img_name)
    image = Image.open(img_file).convert("RGB")
    draw = ImageDraw.Draw(image)

    xml = open(os.path.join(ann_root, xml_file), "r")
    tree = Et.parse(xml)
    root = tree.getroot()

    size = root.find("size")

    width = size.find("width").text
    height = size.find("height").text
    channels = size.find("depth").text

    objects = root.findall("object")

    for _object in objects:
        name = _object.find("name").text
        bndbox = _object.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Box를 그릴 때, 왼쪽 상단 점과, 오른쪽 하단 점의 좌표를 입력으로 주면 됩니다.
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red")
        draw.text((xmin, ymin), name, font = font, fill=(255,0,0,255))

    plt.figure(figsize=(15,10))
    plt.imshow(image)
    plt.savefig('./output_temp/output_'+os.path.splitext(img_name)[0]+'.jpg')
    
    print(count,"/",n_plot,"converted!!!")

    # plt.imshow(image)
    # plt.show()
    # plt.close()
    
    if count % n_plot == 0:
        sys.exit()



