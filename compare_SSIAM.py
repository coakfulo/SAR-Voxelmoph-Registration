
from skimage.measure import compare_ssim as ssim
import cv2
import matplotlib.pyplot as plt


def compare_images(imageA, imageB, title):
    # 分别计算输入图片的MSE和SSIM指标值的大小
    s = ssim(imageA, imageB)
    #return s

    # 创建figure
    fig = plt.figure(title)
    plt.suptitle("SSIM: %.2f" % (s))

    # 显示第一张图片
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")

    # 显示第二张图片
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
