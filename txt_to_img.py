import torch
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F
import PIL.Image as Image
import matplotlib.pyplot as plt


def save_coordinate(mode, index, count, flowpath, savedir):
    csvpath = '../../data/My/{}Images/{}corr.csv'.format(mode.capitalize(), mode.capitalize())
    df = pd.read_csv(csvpath, dtype=str)

    Ispath = '../' + df['source_path'][index]
    Itpath = '../' + df['target_path'][index]
    print('\t--> name of target image: ', Itpath.split('/')[-1])
    print('\tshape of target image(w, h)：', targetsize, targetsize)

    Is = Image.open(Ispath)
    width, height = Is.size
    Is = np.array(Is)
    print('\t--> name of source image: ', Ispath.split('/')[-1])
    print('\tshape of source image(w, h)：', width, height)

    # 读取文件名中左上角初始坐标
    Isname = Ispath.replace('.png', '').split('/')[-1]
    Is_upleft_x = float(Isname.split('_')[-2])
    Is_upleft_y = float(Isname.split('_')[-1])

    Itname = Itpath.replace('.png', '').split('/')[-1]
    It_upleft_x = float(Itname.split('_')[-2])
    It_upleft_y = float(Itname.split('_')[-1])

    for i in range(count):
        print('\tdeal with coordinate[{}]'.format(i + 1))
        It = np.zeros((targetsize, targetsize))
        with open('{}/{}_{}_{}.txt'.format(savedir, index, i + 1, Isname.split('_')[1]), 'r') as f:
            tmp = f.readlines()
        for n in tmp:
            x1, y1, x2, y2 = map(float, n[:-1].split(' '))
            x1 = round(x1 - Is_upleft_x)
            y1 = round(y1 - Is_upleft_y)
            x2 = round(x2 - It_upleft_x)
            y2 = round(y2 - It_upleft_y)
            It[x2][y2] = Is[x1][y1]
        It = list(zip(*It))
        plt.figure()
        plt.axis('off')
        plt.imshow(It)
        plt.show()


if __name__ == '__main__':
    targetsize = 512
    mode = 'train'

    # flowdir = 'My_Stage3_{}_Fine'.format(mode)
    # savedir = flowdir + '_txt'
    # os.makedirs(savedir, exist_ok=True)
    # for flowname in os.listdir(flowdir):
    #     if flowname.startswith('flow_'):
    #         flowpath = os.path.join(flowdir, flowname)
    #         index = int(flowname.split('_')[1])
    #         count = int(flowname.split('_')[-1].replace('H.npy', ''))
    #         print('----- {} image [{}] -----'.format(mode.capitalize(), index))
    #         save_coordinate(mode, index, count, flowpath, savedir)

    flowdir = 'My_Stage3_{}_Fine'.format(mode)
    savedir = flowdir + '_txt'
    os.makedirs(savedir, exist_ok=True)
    # flowpath = os.path.join(flowdir, 'flow_1_1H.npy')
    flowpath = os.path.join(flowdir, 'flow_12_1H.npy')
    index = 12
    count = 1
    save_coordinate(mode, index, count, flowpath, savedir)



