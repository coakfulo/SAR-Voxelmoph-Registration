# -*- coding:utf8 -*-
 
import os
import shutil
import numpy as np
import pandas as pd
import glob

path_img='/Users/huangwenbin/Desktop/u-net/data/FIRE/Images/'
ls = os.listdir(path_img)
list_len=len(ls)
print('总共图片数目：',list_len)

fixed_dir=path_img
fixed_image_names = glob.glob(os.path.join(fixed_dir, '*.jpg')) # 返回路径下面所有图片的全局路径
#moving_image_names = glob.glob(os.path.join(moving_dir, '*.png'))

# idxs = np.random.randint(0, list_len, size=100)
idxs=range(0,list_len,20)
fixed_data=[]
#moving_data=[]
for idx in idxs:
    fixed_data.append(fixed_image_names[idx])
    #moving_data.append(moving_image_names[idx])
print('抽取图片数：',len(fixed_data))
#print('他们是：',fixed_data)
count=0
for obj in fixed_data:
    print(obj)
    shutil.move(obj,'/Users/huangwenbin/Desktop/u-net/data/FIRE/my/')
    count+=1
print('成功数目：',count)