"""
data generators for VoxelMorph

for the CVPR and MICCAI papers, we have data arranged in train/validate/test folders
inside each folder is a /vols/ and a /asegs/ folder with the volumes
and segmentations. All of our papers use npz formated data.
"""

import os, sys
import numpy as np
import cv2



def vxm_data_generator(x_data, batch_size=32):
    """
    generator that takes in data of size [N, H, W], and yields data for our vxm model
    
    Note that we need to provide numpy data for each input, and each output
    
    inputs:  moving_image [bs, H, W, 1], fixed_image [bs, H, W, 1]
    outputs: moved_image  [bs, H, W, 1], zeros [bs, H, W, 2]
    """
    # preliminary sizing
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation. We'll explain this below.
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # prepare inputs
        # inputs need to be of the size [batch_size, H, W, number_features]
        #   number_features at input is 1 for us
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        
        # outputs
        # we need to prepare the "true" moved image.  
        # Of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = [fixed_images, zero_phi]
        
        yield inputs, outputs  
# # 自定义图片生成器
# # 读取图片函数
def get_im_cv2(paths, img_rows, img_cols, color_type=1, normalize=True):
    '''
    参数：
        paths：要读取的图片路径列表
        img_rows:图片行
        img_cols:图片列
        color_type:图片颜色通道
    返回: 
        imgs: 图片数组
    '''
    # Load as grayscale
    imgs = []
    for path in paths:
        if color_type == 1:
            img = cv2.imread(path, 0)
        elif color_type == 3:
            img = cv2.imread(path)
        # Reduce size
        resized = cv2.resize(img, (img_cols, img_rows))
        if normalize:
            resized = resized.astype('float32')
            resized /= 255
            resized -= 1.
        
        imgs.append(resized)
        
    return np.array(imgs).reshape(len(paths), img_rows, img_cols, color_type)

def my_data_generator(x_train_dirs, y_train_dirs, batch_size=32,img_size=480,color_type=1):
    """
    generator that takes in data of size [N, H, W], and yields data for our vxm model
    
    Note that we need to provide numpy data for each input, and each output
    
    inputs:  moving_image [bs, H, W, 1], fixed_image [bs, H, W, 1]
    outputs: moved_image  [bs, H, W, 1], zeros [bs, H, W, 2]
    """
    list_len=len(x_train_dirs)
    vol_shape = (img_size,img_size) # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation. 
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    img_w, img_h=img_size,img_size
    while True:

        idxs = np.random.randint(0, list_len, size=batch_size)
        fixed_data=[]
        moving_data=[]
        for idx in idxs:
            fixed_data.append(x_train_dirs[idx])
            moving_data.append(y_train_dirs[idx])
        moving_images = get_im_cv2(moving_data,img_w,img_h,color_type)
        fixed_images = get_im_cv2(fixed_data,img_w,img_h,color_type)

        inputs = [moving_images, fixed_images]
        outputs = [fixed_images, zero_phi]
        
        yield inputs, outputs  
