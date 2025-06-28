import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from tifffile import TiffFile


def draw_histgram():
    """
    该方法用来绘制柱状图
    :return:
    """
    x_data = ['CC->MI', 'CC->CY', 'HSIL->MI', 'HSIL->CY', 'HSIL->EP', 'MI->HSIL', 'CY->HSIL', 'EP->HSIL']
    data = [2, 2, 3, 4, 6, 4, 4, 9]
    # 设置字体为新罗马字体
    plt.rc('font', family='Times New Roman')
    # 设置y坐标轴范围
    plt.figure(figsize=(7.5, 4))
    plt.ylim((0, 10))
    color_config = ['#91CAE8', '#91CAE8', '#91CAE8', '#91CAE8', '#91CAE8',
                    '#F48892', '#F48892', '#F48892']
    # plt.bar(x1_data, data1, width=0.5, fc='deepskyblue')
    for i in range(len(x_data)):
        if i == 0:
            plt.bar(x_data[i], data[i], width=0.5, color=color_config[i], label='false negative')
        elif i == 5:
            plt.bar(x_data[i], data[i], width=0.5, color=color_config[i], label='false postive')
        else:
            plt.bar(x_data[i], data[i], width=0.5, color=color_config[i])
    plt.legend(loc='upper left', fontsize=12)

    # plt.bar(x_data, data, width=0.5, color=color_config, alpha=1.0)
    # # 显示数据标签
    for a, b in zip(x_data, data):
        plt.text(a, b,
                 b,
                 ha='center',
                 va='bottom',
                 )
    # 设置x坐标轴名称和y坐标轴名称
    # plt.xlabel('False Negative Type')
    plt.xlabel('Misclassification Type', fontsize=14)
    plt.savefig('misclassification.pdf', dpi=1200, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    draw_histgram()
