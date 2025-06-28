import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from util import create_directory
from matplotlib.colors import LinearSegmentedColormap

mode = ['val', 'fine-tune_val']
model_name = ['lbp_resnet101', 'resnet101']
classes_name = ['MI', 'CY', 'EP', 'HSIL', 'CC']
bin_classes_name = ['N', 'P']
title_names = ['model', 'investigator1', 'investigator2', 'investigator3', 'investigator4']


def trans_color(hex_color):
    # 将16进制颜色转换为RGB格式
    rgb_color = tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))

    # 创建自定义的colormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", [rgb_color, rgb_color])
    return custom_cmap


def show_confMat(confusion_mat, label_name, dataset_name, label_names, atttach_pred):
    """
    this function is used to visualize the confusion matrix
    :param confusion_mat: the confusion matrix
    :param label_name: 'a'
    :return:
    """
    # 归一化
    fig, ax = plt.subplots()
    plt.title('(' + label_name + ')', y=-0.2, fontweight='black', fontdict={'fontsize': 20})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # 根据label_name判断class_name
    # class_name = classes_name if label_name <= 'e' else bin_classes_name
    class_name = bin_classes_name
    confusion_mat_N = np.zeros((confusion_mat.shape[0], confusion_mat.shape[1]))
    for i in range(len(class_name)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    # color = 'Blues' if label_name in ['a', 'f'] else 'Reds'
    color = 'Greens' if label_name in label_names[:2] else 'Oranges'
    # color = 'Purples' if label_name in label_names[:2] else 'GnBu'
    cmap = plt.cm.get_cmap(color)  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    # 只有label name 为 'a', 'e', 'f', 'j'时才进行colorbar的绘制
    if label_name in [label_names[1], label_names[-1]]:
        cb1 = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=7)  # colorbar上的刻度值个数
        cb1.locator = tick_locator
        # cb1.set_ticks([0.0, 0.2, 0.4])
        cb1.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1])
        cb1.ax.tick_params(labelsize=14)
        cb1.update_ticks()

    # 设置文字
    xlocations = np.array(range(len(class_name)))
    plt.xticks(xlocations, class_name, fontsize=14)
    plt.yticks(xlocations, class_name, fontsize=14)
    if atttach_pred:
        plt.xlabel('Prediction', fontsize=14, labelpad=0)
    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            if confusion_mat_N[i][j] > 0.2:
                plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='white', fontsize=24)
            else:
                plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='black', fontsize=24)
    # 保存
    # plt.show()
    data_dir = os.path.join('confusion_matrix_image', dataset_name)
    create_directory(data_dir)

    # plt.savefig(os.path.join(data_dir, label_name + '.svg'), bbox_inches='tight', dpi=1200)
    # 保存图像为 PDF 格式
    plt.savefig(os.path.join(data_dir, label_name + '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()


def draw_conf_matrix(dataset_name, conf_matrix_list,
                     name_list, atttach_pred):
    for conf_matrix, label_name in zip(conf_matrix_list, name_list):
        show_confMat(confusion_mat=conf_matrix, label_name=label_name,
                     dataset_name=dataset_name, label_names=name_list,
                     atttach_pred=atttach_pred)


if __name__ == '__main__':
    name_list = ['a', 'b', 'c', 'd', 'e', 'f']
    conf_matrix_list = []
    convnext_pam = np.array([[665, 14], [8, 73]])
    convnext = np.array([[663, 16], [10, 71]])
    inv1 = np.array([[678, 1], [10, 71]])
    inv2 = np.array([[679, 0], [12, 69]])
    inv3 = np.array([[664, 15], [12, 69]])
    inv4 = np.array([[660, 19], [12, 69]])
    conf_matrix_list.append(convnext_pam)
    conf_matrix_list.append(convnext)
    conf_matrix_list.append(inv1)
    conf_matrix_list.append(inv2)
    conf_matrix_list.append(inv3)
    conf_matrix_list.append(inv4)
    draw_conf_matrix('huaxi', conf_matrix_list, name_list, atttach_pred=False)

    name_list = ['g', 'h', 'i', 'j', 'k', 'l']
    conf_matrix_list = []
    convnext_pam = np.array([[216, 2], [5, 55]])
    convnext = np.array([[217, 1], [7, 53]])
    inv1 = np.array([[218, 0], [6, 54]])
    inv2 = np.array([[213, 5], [6, 54]])
    inv3 = np.array([[216, 2], [10, 50]])
    inv4 = np.array([[216, 2], [11, 49]])
    conf_matrix_list.append(convnext_pam)
    conf_matrix_list.append(convnext)
    conf_matrix_list.append(inv1)
    conf_matrix_list.append(inv2)
    conf_matrix_list.append(inv3)
    conf_matrix_list.append(inv4)
    draw_conf_matrix('xiangya', conf_matrix_list, name_list, atttach_pred=True)
