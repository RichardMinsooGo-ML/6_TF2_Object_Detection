"""
import os

for root, dirs, files in os.walk("./mnist_test"):
    for filename in files:
        print(filename)
        
"""        

import os

str_train = open("train.txt", "w")
str_test = open("val.txt", "w")

for path, subdirs, files in os.walk(r"./xml_train"):
    for filename in files:
        # f = os.path.join(path, filename)
        f = os.path.join(filename)
        f = os.path.splitext(f)[0]
        
        str_train.write(str(f) + os.linesep)
        # str_train.write(str(f))

        
for path, subdirs, files in os.walk(r"./xml_val"):
    for filename in files:
        # f = os.path.join(path, filename)
        f = os.path.join(filename)
        f = os.path.splitext(f)[0]
        
        str_test.write(str(f) + os.linesep)
        # str_test.write(str(f))

"""
count = 0
rand_remain = 3   # choose between 1 ~ 13

for path, subdirs, files in os.walk(r"./JPEGImages"):
    for filename in files:
        if count > 1250*13:
            import sys
            sys.exit()

        # f = os.path.join(path, filename)
        f = os.path.join(filename)
        f = os.path.splitext(f)[0]
        count += 1
        
        select_file = divmod(count, 13)
        
        if select_file[1] == rand_remain:
            if (select_file[0]+1)%5 != 0:
                str_train.write(str(f) + os.linesep)
                # str_train.write(str(f))
            else:
                str_test.write(str(f) + os.linesep)
                # str_test.write(str(f))
        else:
            pass
"""        


