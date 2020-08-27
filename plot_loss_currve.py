import pickle
import matplotlib.pyplot as plt


with open('data/SAR_model_hist.pickle','rb') as file_pi:   
    hist=pickle.load(file_pi)

def plot_history(hist, loss_name='loss',path='data/'):

    """
    Quick function to plot the history 
    """
    plt.figure()
    plt.plot(hist[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(loss_name)
    plt.savefig(path)
    plt.show()

plot_history(hist,'loss','data/loss.jpg')
plot_history(hist,'val_loss','data/val_loss.jpg')
plot_history(hist,'disp_loss','data/disp_loss.jpg')
plot_history(hist,'val_disp_loss','data/val_disp_loss.jpg')
plot_history(hist,'spatial_transformer_loss','data/spatial_transformer_loss.jpg')
plot_history(hist,'val_spatial_transformer_loss','data/val_spatial_transformer_loss.jpg')
print(hist.keys())
