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

# 添加voxelmorph支持包路径
sys.path.append('/Users/huangwenbin/Desktop/SAR-voxelmorph/ext/pynd-lib/')
sys.path.append('/Users/huangwenbin/Desktop/SAR-voxelmorph/ext/pytools-lib/')
sys.path.append('/Users/huangwenbin/Desktop/SAR-voxelmorph/ext/neuron/')
import neuron


# 添加本地函数
import networks
import losses
import datagenerators
import flow_smooth
os.environ['KMP_DUPLICATE_LIB_OK']='True' # 避免macos 报错OMP
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def test(root_dir,fixed_dir,moving_dir,model_dir,gpu_id,img_size):
    #
    # 定义模型，加载权重
    # 输入维度
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
    #loss = ['mse', losses.Grad('l2').loss]
    loss = [losses.NCC().loss, losses.Grad('l2').loss]

    # 损失函数
    lambda_param = 0.01
    loss_weights = [1, lambda_param]
    
    #---------------加载模型权重-------------------------
    vxm_model.compile(optimizer='Adam', loss=loss, loss_weights=loss_weights)
    vxm_model.load_weights(model_dir)
    
    #--------------前向推理------------------------------------
    fixed_vol_names = glob.glob(os.path.join(fixed_dir, '*.png'))
    moving_vol_names = glob.glob(os.path.join(moving_dir, '*.png'))
    print(fixed_vol_names,moving_vol_names)
    # fixed_vol_names.sort()
    # moving_vol_names.sort()
    data=datagenerators.my_data_generator(fixed_vol_names,moving_vol_names, batch_size=1,img_size=img_size)
    sample, _ = next(data)
    print('输入维度：',sample[0].shape)
    sample_pred = vxm_model.predict(sample)

    #---------------保存流场维度数据------------------------------
    print('流场输出维度：',sample_pred[1].shape)
    #a=sample_pred[0].squeeze()
    slices_in=[sample_pred[1].squeeze()]
    np.save(root_dir+'flow.npy',slices_in)
    u, v = slices_in[0][...,0], slices_in[0][...,1]
    #u, v = flow_smooth.smooth(u,v,1)
    # np.savetxt(root_dir+'a.csv',a,delimiter=",")
    np.savetxt(root_dir+'u.csv',u,delimiter=",")
    np.savetxt(root_dir+'v.csv',v,delimiter=",")      # 保存偏移值

    #------------输出配准点的列表-------txt---------------------------
    txt=open(root_dir+'match.txt','w')
    moving_img=Image.open(moving_vol_names[0])
    # moving_pixeles=str(moving_vol_names[0]).split('_')
    # fixed_pixeles=str(fixed_vol_names[0]).split('_')

    # print(moving_pixeles)
    # print(fixed_pixeles)

    # x_moving=float(moving_pixeles[2])
    # y_moving=float(moving_pixeles[3].strip('.png'))  # 读取文件名中左上角初始坐标
    # x_fixed=float(fixed_pixeles[2])
    # y_fixed=float(fixed_pixeles[3].strip('.png'))

    # width,height=moving_img.size
    # print('待配准图片长宽：',width,height)
    # w_scale=width/img_size
    # h_scale=height/img_size
    # for i in range(img_size):
    #     for j in range(img_size):
    #         x1=str(i*w_scale+x_moving)     # 坐标转换
    #         y1=str(j*h_scale+y_moving)
    #         x2=str(i+u[i][j]+x_fixed)
    #         y2=str(j-v[i][j]+y_fixed)
    #         txt.write(x1+' '+y1+' '+x2+' '+y2)
    #         txt.write('\n')
    width,height=moving_img.size
    print('待配准图片长宽：',width,height)
    w_scale=width/img_size
    h_scale=height/img_size
    for i in range(img_size):
        for j in range(img_size):
            # if  u[i][j]!=0:
            x1=str(i*w_scale)     # 坐标转换
            y1=str(j*h_scale)
            x2=str(i+u[i][j])
            y2=str(j-v[i][j])
            txt.write(x1+' '+y1+' '+x2+' '+y2)
            txt.write('\n')

    #---------------可视化配准图像和流场-----------------------------
    slices_2d = [f[0,...,0] for f in sample + sample_pred]
    # fixed=sample[1].squeeze()
    # moving=sample[0].squeeze()
    warped=sample_pred[0].squeeze()

    #print(warped.shape)
    plt.imshow(warped, cmap='Greys_r')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.margins(0,0)
    plt.axis('off')
    plt.savefig(root_dir+'warped_picture.png',transparent=True,dpi=300,pad_inches=0.0)
    # plt.show()
 
    titles = ['input_moving', 'input_fixed', 'predicted_moved', 'deformation_x']
    neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True,path=root_dir+'predict.png');
    neuron.plot.flow([sample_pred[1].squeeze()], width=5,path=root_dir+'flow.png');


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--root_dir", type=str,default='data/test2/',
                        help="结果输出目录")
    parser.add_argument("--fixed_dir", type=str,default='data/test2/fixed/',
                        help="固定图像目录")
    parser.add_argument("--moving_dir", type=str,default='data/test2/moving/',
                        help="待配准图像目录")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='models/TR1-MSE-300.h5',
                        help="models folder")
    parser.add_argument("--gpu", type=str, default='0',
                        dest="gpu_id", help="gpu id number (or numbers separated by comma)")
    parser.add_argument("--img_size", type=int,
                        default=512,
                        help="image size")

    args = parser.parse_args()
    print('测试参数：',args)
    test(**vars(args))