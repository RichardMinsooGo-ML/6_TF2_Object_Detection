import os

# Delete the contents at the directories = remove directory if exist
# dirpath = os.path.join('dataset3', 'dataset')
dirpath = "txt_train"
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

dirpath = "txt_val"
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)
    
    
# Create empty folder
dirpath = "txt_train"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)
    
dirpath = "txt_val"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)
    

filename = './tmp_mnist_train.txt'
with open(filename) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 


print(len(content))

# print(content)
import re
import numpy as np

# Swap function 
def swapPositions(list, pos1, pos2): 
      
    list[pos1], list[pos2] = list[pos2], list[pos1] 
    return list

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '%.12f' % f
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

for row_idx in range(len(content)):
    # Driver code
    
    a_string = content[row_idx]
    split_string = a_string.replace(' ',',').split(',')
    print(split_string)
    
    # print(len(split_string))
    # print(int(len(split_string)/5))

    for idx in range(int(len(split_string)/5)):
        swapPositions(split_string, idx*5 + 6-1, idx*5 + 5-1)
        swapPositions(split_string, idx*5 + 5-1, idx*5 + 4-1)
        swapPositions(split_string, idx*5 + 4-1, idx*5 + 3-1)
        swapPositions(split_string, idx*5 + 3-1, idx*5 + 2-1)
    # print(split_string,"\n")


    for idx in range(int(len(split_string)/5)):
        val_2 = float(split_string[idx*5 + 2])/416
        val_2 = truncate(val_2, 6)
        split_string[idx*5 + 2] = str(val_2) 
        
        val_3 = float(split_string[idx*5 + 3])/416
        val_3 = truncate(val_3, 6)
        split_string[idx*5 + 3] = str(val_3) 
        
        val_4 = float(split_string[idx*5 + 4])/416
        val_4 = truncate(val_4, 6)
        split_string[idx*5 + 4] = str(val_4) 
        
        val_5 = float(split_string[idx*5 + 5])/416
        val_5 = truncate(val_5, 6)
        split_string[idx*5 + 5] = str(val_5) 

    print(split_string,"\n")

    with open("./txt_train/"+split_string[0]+".txt", "w") as output:
        save_content = " ".join(split_string[1:len(split_string)])
        output.write(save_content)    

        
filename = './tmp_mnist_val.txt'
with open(filename) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

print(len(content))

# print(content)
import re
import numpy as np

# Swap function 
def swapPositions(list, pos1, pos2): 
      
    list[pos1], list[pos2] = list[pos2], list[pos1] 
    return list

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '%.12f' % f
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

for row_idx in range(len(content)):
    # Driver code
    
    a_string = content[row_idx]
    split_string = a_string.replace(' ',',').split(',')
    print(split_string)
    
    # print(len(split_string))
    # print(int(len(split_string)/5))

    for idx in range(int(len(split_string)/5)):
        swapPositions(split_string, idx*5 + 6-1, idx*5 + 5-1)
        swapPositions(split_string, idx*5 + 5-1, idx*5 + 4-1)
        swapPositions(split_string, idx*5 + 4-1, idx*5 + 3-1)
        swapPositions(split_string, idx*5 + 3-1, idx*5 + 2-1)
    # print(split_string,"\n")


    for idx in range(int(len(split_string)/5)):
        val_2 = float(split_string[idx*5 + 2])/416
        val_2 = truncate(val_2, 6)
        split_string[idx*5 + 2] = str(val_2) 
        
        val_3 = float(split_string[idx*5 + 3])/416
        val_3 = truncate(val_3, 6)
        split_string[idx*5 + 3] = str(val_3) 
        
        val_4 = float(split_string[idx*5 + 4])/416
        val_4 = truncate(val_4, 6)
        split_string[idx*5 + 4] = str(val_4) 
        
        val_5 = float(split_string[idx*5 + 5])/416
        val_5 = truncate(val_5, 6)
        split_string[idx*5 + 5] = str(val_5) 

    print(split_string,"\n")

    with open("./txt_val/"+split_string[0]+".txt", "w") as output:
        save_content = " ".join(split_string[1:len(split_string)])
        output.write(save_content)    


