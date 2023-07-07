
# -*- coding:utf-8 -*-
 
import os
import shutil
 
src_path = "./wait_to_be_trainset"
dst_path = "./wait_to_be_trainset_noclass"
 
def mycopy(srcpath, dstpath):
    if not os.path.exists(srcpath):
        print("srcpath not exist!")
    if not os.path.exists(dstpath):
        print("dstpath not exist!")
    for root, dirs, files in os.walk(srcpath, True):
        for eachfile in files:
            shutil.copy(os.path.join(root, eachfile), dstpath)
 
if __name__ == "__main__":
    mycopy(src_path, dst_path)