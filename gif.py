import imageio 

def creat_gif(image_list, gif_name, duration = 2):
    """
    生成gif文件，原始图像仅仅支持png格式；
    gif_name : 字符串，所生成的gif文件名，带.gif文件名后缀；
    path : 输入图像的路径；
    duration : gif图像时间间隔，这里默认设置为1s,当然你喜欢可以设置其他；
    """
    # 创建一个空列表，用来存源图像
    frames = []
    
    # 利用方法append把图片挨个存进列表
    
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
 
    # 保存为gif格式的图
    imageio.mimsave(gif_name, frames, 'GIF', duration = duration)
 
    return
    
def main():
    # 自己收集的图片，当然这里也可以爬取网上的图像，加载进来
    image_list = ['data/gif/1/基准_vs_待配准4.png', 'data/gif/1/_基准vs_配准后4.png'] # 源图像，可以随便选取哈
    gif_name = 'data/gif/1/new.gif'
    duration = 0.8
    creat_gif(image_list, gif_name,duration)
if __name__ == "__main__":
   main()