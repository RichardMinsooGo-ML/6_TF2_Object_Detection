import os
import shutil

dirpath = "txt_train"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)
    
dirpath = "txt_test"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)
    
# dirpath = os.path.join('dataset3', 'dataset')
dirpath = "txt_train"
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

dirpath = "txt_test"
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)



