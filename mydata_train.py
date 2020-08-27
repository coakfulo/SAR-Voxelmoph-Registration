# imports
import os
import sys
import glob
from argparse import ArgumentParser
import pickle

# 导入包
import numpy as np
import keras.layers
from keras.optimizers import Adam
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint,TensorBoard
import matplotlib.pyplot as plt

# 添加voxelmorph支持包路径
sys.path.append('/Users/huangwenbin/Desktop/SAR-voxelmorph/ext/pynd-lib/')
sys.path.append('/Users/huangwenbin/Desktop/SAR-voxelmorph/ext/pytools-lib/')
sys.path.append('/Users/huangwenbin/Desktop/SAR-voxelmorph/ext/neuron/')
sys.path.append('/usr/local/lib/python3.7/site-packages')
import neuron

os.environ['KMP_DUPLICATE_LIB_OK']='True' # 避免macos 报错OMP
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 添加本地函数
import networks
import losses
import datagenerators

def train(fixed_dir, 
          moving_dir,
          fixedval_dir, 
          movingval_dir,
          model_dir,
          gpu_id,
          lr,
          nb_epochs,
          img_size,
          lambda_param,
          steps_per_epoch,
          batch_size,
          ):

    fixed_training_file=open(fixed_dir,'r')
    moving_traning_file=open(moving_dir,'r')
    fixed_val_file=open(fixedval_dir,'r')
    moving_val_file=open(movingval_dir,'r')

    def str_strip(file):
        list=[]
        for f in file.readlines():
            list.append(f.strip())
        return list
    fixed_image_names = str_strip(fixed_training_file)
    moving_image_names = str_strip(moving_traning_file)  
    fixedval_image_names = str_strip(fixed_val_file)
    movingval_image_names = str_strip(moving_val_file)
    assert len(fixed_image_names) > 0, "fixed路径中找不到训练数据"
    assert len(moving_image_names) > 0, "moving路径中找不到训练数据"
    assert len(fixedval_image_names) > 0, "fixed路径中找不到训练数据"
    assert len(movingval_image_names) > 0, "moving路径中找不到训练数据"

    # GPU handling
    gpu = '/gpu:%d' % 0 # gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    #-------------------------------模型定义-----------------------------------     
    # 输入维度
    ndims = 2
    vol_shape = (img_size,img_size)
    nb_enc_features = [32, 32, 32, 32]  # 下采样卷积核个数
    nb_dec_features = [32, 32, 32, 32, 32, 16]  # 上采样卷积核个数

    # 网络定义U-net
    unet = networks.unet_core(vol_shape, nb_enc_features, nb_dec_features);

    #------网络输入--------------------
    print('网络输入维度', len(unet.inputs))
    moving_input_tensor = unet.inputs[0]
    fixed_input_tensor = unet.inputs[1]
        
    #---------网络输出----------------
    print('网络输出维度:', unet.output)

    #-------------转换成流场维度---------------
    disp_tensor = keras.layers.Conv2D(ndims, kernel_size=3, padding='same', name='disp')(unet.output)
    # （512，512，2）
    # check
    print('displacement tensor:', disp_tensor)

    #--------------------流场作用于原图扭曲-------------------
    spatial_transformer = neuron.layers.SpatialTransformer(name='spatial_transformer')
    moved_image_tensor = spatial_transformer([moving_input_tensor, disp_tensor])

    inputs = [moving_input_tensor, fixed_input_tensor]
    outputs = [moved_image_tensor, disp_tensor]
    vxm_model = keras.models.Model(inputs, outputs)

    # losses. Keras recognizes the string 'mse' as mean squared error, so we don't have to code it
    #loss = [losses.NCC().loss, losses.Grad('l2').loss]
    loss = ['mse', losses.Grad('l2').loss]

    # usually, we have to balance the two losses by a hyper-parameter.
    loss_weights = [1, lambda_param]

    vxm_model.compile(optimizer=Adam(lr=lr), loss=loss, loss_weights=loss_weights)
    vxm_model.summary()

    #----------------------准备训练数据-------------------------------
    x_train_path=fixed_image_names
    y_train_path=moving_image_names
    x_val_path=fixedval_image_names
    y_val_path=movingval_image_names
    train_generator = datagenerators.my_data_generator(x_train_path,y_train_path,batch_size=batch_size,img_size=img_size)
    valid_generator = datagenerators.my_data_generator(x_val_path,y_train_path,batch_size=1,img_size=img_size)
    input_sample, output_sample = next(train_generator)

    # ---------------可视化---------------------
    slices_2d = [f[0,...,0] for f in input_sample + output_sample]
    titles = ['input_moving', 'input_fixed', 'output_moved_ground_truth', 'zero']
    neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);

    checkpoint = ModelCheckpoint(filepath='models/my_model{epoch:03d}.h5', monitor='loss',verbose=1,save_best_only='True',mode='min',period=1)
    tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, write_graph=True, write_images=True)
    #-------------------------开始训练---------------------------
    hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2,callbacks=[checkpoint,tensorboard],validation_data=valid_generator,validation_steps=2990);
    vxm_model.save_weights(os.path.join(model_dir, 'my_model.h5' ))
    with open('data/SAR_model_hist.pickle', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)

    def plot_history(hist, loss_name='loss',path='data/'):
        """
        Quick function to plot the history 
        """
        plt.figure()
        plt.plot(hist.epoch, hist.history[loss_name], '.-')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig(path)
        plt.show()

    plot_history(hist,'loss','data/loss.jpg')
    plot_history(hist,'val_loss','data/val_loss.jpg')
    #----------------直接测试一张图配准--------------------------
    val_generator = datagenerators.my_data_generator(x_train_path,y_train_path, batch_size = 1,img_size=img_size)
    val_input, _ = next(val_generator)
    val_pred = vxm_model.predict(val_input)

    # ---------------可视化---------------------
    slices_2d = [f[0,...,0] for f in val_input + val_pred]
    titles = ['input_moving', 'input_fixed', 'predicted_moved', 'deformation_x']
    neuron.plot.slices(slices_2d, titles=titles, cmaps=['gray'], do_colorbars=True);
    neuron.plot.flow([val_pred[1].squeeze()], width=5);

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--fixed_dir", type=str,default='/Users/huangwenbin/Desktop/SAR-voxelmorph/data/SAR_CUTS/SARcuts2/txt/train_fixed.txt',
                        help="data folder")
    parser.add_argument("--moving_dir", type=str,default='/Users/huangwenbin/Desktop/SAR-voxelmorph/data/SAR_CUTS/SARcuts2/txt/train_moving.txt',
                        help="data folder")
    parser.add_argument("--fixedval_dir", type=str,default='/Users/huangwenbin/Desktop/SAR-voxelmorph/data/SAR_CUTS/SARcuts2/txt/val_fixed.txt',
                        help="data folder")
    parser.add_argument("--movingval_dir", type=str,default='/Users/huangwenbin/Desktop/SAR-voxelmorph/data/SAR_CUTS/SARcuts2/txt/val_moving.txt',
                        help="data folder")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='models/',
                        help="models folder")
    parser.add_argument("--gpu", type=str, default='0',
                        dest="gpu_id", help="gpu id number (or numbers separated by comma)")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=3,
                        help="number of iterations")
    parser.add_argument("--img_size", type=int,default=32,
                        help="image size")
    parser.add_argument("--ambda_param", type=float,
                        dest="lambda_param", default=0.01,  # recommend 1.0 for ncc, 0.01 for mse
                        help="regularization parameter")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=19943,
                        help="frequency of model saves")
    parser.add_argument("-- batch_size", type=int,
                        dest="batch_size", default=1,
                        help="batch_size")

    args = parser.parse_args()
    print('训练参数：',args)
    train(**vars(args))
