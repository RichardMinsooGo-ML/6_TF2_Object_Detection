import os
import shutil

dirpath = "txt_train"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)
    
dirpath = "txt_val"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)
    
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

