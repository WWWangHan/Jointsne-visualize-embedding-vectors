import numpy as np
import warnings
warnings.filterwarnings('ignore')

# for painting
import Yoked_Tsne, Jointsne, Scatter


def visualize(dataset_name, paint_epoch):
    
    all_paint_fea_train = []
    all_paint_class_train = []
    all_paint_fea_val = []
    all_paint_class_val = []

    paint_fea_train = np.load("npy/epoch_{}/paint_fea_train_{}_epoch_{}.npy".format(str(paint_epoch), str(dataset_name), str(paint_epoch)), allow_pickle=True)
    paint_class_train = np.load("npy/epoch_{}/paint_class_train_{}_epoch_{}.npy".format(str(paint_epoch), str(dataset_name), str(paint_epoch)), allow_pickle=True)
    paint_fea_val = np.load("npy/epoch_{}/paint_fea_val_{}_epoch_{}.npy".format(str(paint_epoch), str(dataset_name), str(paint_epoch)), allow_pickle=True)
    paint_class_val = np.load("npy/epoch_{}/paint_class_val_{}_epoch_{}.npy".format(str(paint_epoch), str(dataset_name), str(paint_epoch)), allow_pickle=True)

    is_batch = False
    pic_save_path = 'npy/epoch_{}/'.format(str(paint_epoch))
    

    if is_batch:
        batch_num2_visualize = 0  # to visualize which batch
        batch_pic_name_train = 'scatter_plot_train_batch_{}'.format(str(batch_num2_visualize))    
        batch_pic_name_val = 'scatter_plot_val_batch_{}'.format(str(batch_num2_visualize))
        
        Jointsne.jointTsne(paint_fea_train[batch_num2_visualize], paint_class_train[batch_num2_visualize],
                           paint_fea_val[batch_num2_visualize], paint_class_val[batch_num2_visualize], path=pic_save_path, name='tog_batch')  # together
        Scatter.gene_sca(paint_fea_train[batch_num2_visualize], paint_class_train[batch_num2_visualize], name=batch_pic_name_train, savepath=pic_save_path)  # train
        Scatter.gene_sca(paint_fea_val[batch_num2_visualize], paint_class_val[batch_num2_visualize], name=batch_pic_name_val, savepath=pic_save_path)  # val
    else:
        # deal with training set
        for i in range(len(paint_fea_train)):
            for j in range(len(paint_fea_train[i])):
                all_paint_fea_train.append(paint_fea_train[i][j])
                all_paint_class_train.append(paint_class_train[i][j])
        # deal with testing set
        for k in range(len(paint_fea_val)):
            for m in range(len(paint_fea_val[k])):
                all_paint_fea_val.append(paint_fea_val[k][m])
                all_paint_class_val.append(paint_class_val[k][m])

        Jointsne.jointTsne(all_paint_fea_train, all_paint_class_train, all_paint_fea_val, all_paint_class_val, path=pic_save_path)  # together
        print("Jointsne painting finished.")
        Scatter.gene_sca(all_paint_fea_train, all_paint_class_train, name='scatter_plot_train', savepath=pic_save_path)  # train
        print('Scatter painting for train finished.')
        Scatter.gene_sca(all_paint_fea_val, all_paint_class_val, name='scatter_plot_val', savepath=pic_save_path)  # val
        print('Scatter painting for val finished.')


if __name__ == "__main__":

	dataset_name = "cars196"
	paint_epoch = 0 # 0 or 34

	visualize(dataset_name, paint_epoch)


