import os
import pandas as pd
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

Ispath='/Users/huangwenbin/Desktop/SAR-voxelmorph/data/test2/moving/0_016159-015641_8083.6533_33911.1217.png'
Is = Image.open(Ispath)
print(Is.size)
Is = np.array(Is)    # 待配准图
Is = np.transpose(Is)
print(Is.shape)

targetsize=1024
It = np.zeros((targetsize, targetsize))     # 初始化一个图片
with open('data/test2/match.txt', 'r') as f:
    tmp = f.readlines()
count=0
print(len(tmp))
for n in tmp:
    x1, y1, x2, y2 = map(float, n[:-1].split(' '))
    x1 = round(x1)
    y1 = round(y1)
    x2 = round(x2)
    y2 = round(y2)
    try:
        It[x2][y2] = Is[x1][y1]
    except:
        print(x1, y1, x2, y2)
        count+=1
print('在图片外点数目：',count)
It = list(zip(*It))
plt.figure()
plt.axis('off')
plt.imshow(It,cmap='gray')
plt.show()