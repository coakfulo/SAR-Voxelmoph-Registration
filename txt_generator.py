import os
import glob

path1='/Users/huangwenbin/Desktop/u-net/data/SAR_CUTS/SARcuts2/target_016159-015323/'
path2='/Users/huangwenbin/Desktop/u-net/data/SAR_CUTS/SARcuts2/target_016159-015467/'
path3='/Users/huangwenbin/Desktop/u-net/data/SAR_CUTS/SARcuts2/target_016159-015568/'
path4='/Users/huangwenbin/Desktop/u-net/data/SAR_CUTS/SARcuts2/target_016159-015641/'
path5='/Users/huangwenbin/Desktop/u-net/data/SAR_CUTS/SARcuts2/target_016159-015712/'


file=open('/Users/huangwenbin/Desktop/u-net/data/SAR_CUTS/SARcuts2/moving.txt','w')
fixed_image_names1 = os.listdir(path1)
fixed_image_names2 = os.listdir(path2)
fixed_image_names3 = os.listdir(path3)
fixed_image_names4 = os.listdir(path4)
fixed_image_names5 = os.listdir(path5)

count=0
for xml in sorted(fixed_image_names1):
    file.write(path1+xml+'\n')
    count+=1
for xml in sorted(fixed_image_names2):
    file.write(path2+xml+'\n')
    count+=1
for xml in sorted(fixed_image_names3):
    file.write(path3+xml+'\n')
    count+=1
for xml in sorted(fixed_image_names4):
    file.write(path4+xml+'\n')
    count+=1
for xml in sorted(fixed_image_names5):
    file.write(path5+xml+'\n')
    count+=1

print('val2处理信息%s条'%(count))
