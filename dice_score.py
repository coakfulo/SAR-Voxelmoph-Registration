# imports
import os
import sys
import glob
from argparse import ArgumentParser
from PIL import Image
import cv2

# third party imports
import numpy as np
import keras.layers
from keras.models import load_model
from keras.optimizers import Adam
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import plot_model
import matplotlib.pyplot as plt
import keras.backend as K

# 添加voxelmorph支持包路径
sys.path.append('/Users/huangwenbin/Desktop/u-net/ext/pynd-lib/')
sys.path.append('/Users/huangwenbin/Desktop/u-net/ext/pytools-lib/')
sys.path.append('/Users/huangwenbin/Desktop/u-net/ext/neuron/')
sys.path.append('/Users/huangwenbin/Desktop/u-net/ext/medipy-lib')
sys.path.append('/usr/local/lib/python3.7/site-packages')
import neuron
from medipy.metrics import dice

# 添加本地函数
import networks
import losses
import datagenerators

os.environ['KMP_DUPLICATE_LIB_OK']='True' # 避免macos 报错OMP
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def test(root_dir,fixed_dir,moving_dir,model_dir,gpu_id,img_size):
    fixed_file=open(fixed_dir,'r')
    moving_file=open(moving_dir,'r')


    def str_strip(file):
        list=[]
        for f in file.readlines():
            list.append(f.strip())
        return list
    fixed_vol_names = str_strip(fixed_file)
    moving_vol_names = str_strip(moving_file)  
    assert len(fixed_vol_names) > 0, "fixed路径中找不到训练数据"
    assert len(moving_vol_names) > 0, "moving路径中找不到训练数据"
    


    #GPU handling
    gpu = '/gpu:%d' % 0 # gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    ndims = 2    
    vol_shape = (img_size,img_size)
    nb_enc_features = [32, 32, 32, 32]  # 下采样卷积核个数
    nb_dec_features = [32, 32, 32, 32, 32, 16]  # 上采样卷积核个数


    # 网络定义U-net
    unet = networks.unet_core(vol_shape, nb_enc_features, nb_dec_features);

    # 输入
    print('numer of inputs', len(unet.inputs))
    moving_input_tensor = unet.inputs[0]
    fixed_input_tensor = unet.inputs[1]

    # 输出
    print('output:', unet.output)

    # 转换为流场维度
    disp_tensor = keras.layers.Conv2D(ndims, kernel_size=3, padding='same', name='disp')(unet.output)

    # 显示流场维度
    print('displacement tensor:', disp_tensor)

    spatial_transformer = neuron.layers.SpatialTransformer(name='spatial_transformer')

    # 扭转图像
    moved_image_tensor = spatial_transformer([moving_input_tensor, disp_tensor])

    inputs = [moving_input_tensor, fixed_input_tensor]
    outputs = [moved_image_tensor, disp_tensor]
    vxm_model = keras.models.Model(inputs, outputs)

    # losses. Keras recognizes the string 'mse' as mean squared error, so we don't have to code it
    loss = ['mse', losses.Grad('l2').loss]

    # 损失函数
    lambda_param = 0.01
    loss_weights = [1, lambda_param]

    #---------------加载模型权重-------------------------
    vxm_model.compile(optimizer='Adam', loss=loss, loss_weights=loss_weights)
    vxm_model.load_weights(model_dir)
    
    #------------定义DICE函数-------------------------------
    def dice_coef(y_true, y_pred):
        y_true_f = y_true.flatten() # 将 y_true 拉伸为一维.
        y_pred_f = y_pred.flatten()
        intersection = sum(y_true_f * y_pred_f)
        return (2. * intersection ) / (sum(y_true_f * y_true_f) + sum(y_pred_f * y_pred_f))

    #--------------前向推理DICE计算------------------------------------
    length=len(fixed_vol_names)
    dice_before=0
    dice_after=0
    for i in range(length):
        data=datagenerators.my_data_generator([fixed_vol_names[i]],[moving_vol_names[i]], batch_size=1,img_size=img_size)
        sample, _ = next(data)
        sample_pred = vxm_model.predict(sample)

        fixed=sample[1].squeeze()
        moving=sample[0].squeeze()
        warped=sample_pred[0].squeeze()

        dice_before+=dice_coef(fixed,moving)
        dice_after+=dice_coef(fixed,warped)

        print(dice_coef(fixed,warped))

    Mean_Dice_befroe=dice_before/length
    Mean_Dice_after=dice_after/length
    print('配准前DICE：',Mean_Dice_befroe)
    print('配准后DICE：',Mean_Dice_after)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--root_dir", type=str,default='data/test/',
                        help="结果输出目录")
    parser.add_argument("--fixed_dir", type=str,default='/Users/huangwenbin/Desktop/u-net/data/SAR_CUTS/SARcuts2/txt/test_fixed.txt',
                        help="固定图像目录")
    parser.add_argument("--moving_dir", type=str,default='/Users/huangwenbin/Desktop/u-net/data/SAR_CUTS/SARcuts2/txt/test_moving.txt',
                        help="待配准图像目录")

    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='models/TR4-COR-200.h5',
                        help="models folder")
    parser.add_argument("--gpu", type=str, default='0',
                        dest="gpu_id", help="gpu id number (or numbers separated by comma)")
    parser.add_argument("--img_size", type=int,
                        default=512,
                        help="image size")

    args = parser.parse_args()
    print('测试参数：',args)
    test(**vars(args))