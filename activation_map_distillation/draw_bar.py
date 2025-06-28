import os
import numpy as np
import matplotlib.pyplot as plt


def barplot():
    plt.rcParams['font.sans-serif'] = ['serif']
    plt.rcParams['axes.unicode_minus'] = False
    title = 'The Volume-Level Classification Performance of Models and Clinical Experts on Xiangya Dataset'
    dir_path = 'bar_plot'
    # file_path = 'Huaxi.pdf'
    file_path = 'Xiangya.pdf'
    fig_path = os.path.join(dir_path, file_path)
    categories = ['Binary Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
    huaxi_ResNet18 = [97.11, 90.12, 97.94, 83.91, 98.81]
    huaxi_ResNet_little = [96.58, 87.65, 97.64, 81.61, 98.51]
    huaxi_Investigator1 = [98.55, 87.65, 99.85, 98.61, 98.55]
    huaxi_Investigator2 = [98.42, 85.19, 100.00, 100.00, 98.26]
    huaxi_Investigator3 = [96.45, 85.19, 97.79, 82.14, 98.22]
    huaxi_Investigator4 = [95.92, 85.19, 97.20, 78.41, 98.21]
    huaxi_Investigator_avg = [97.33, 85.81, 98.71, 89.79, 98.31]
    huaxi_values = np.array([huaxi_ResNet18, huaxi_ResNet_little, huaxi_Investigator1,
                             huaxi_Investigator2, huaxi_Investigator3, huaxi_Investigator4,
                             huaxi_Investigator_avg])  # [7, 5]

    xiangya_ResNet18 = [97.48, 91.67, 99.08, 96.49, 97.74]
    xiangya_ResNet_little = [97.59, 88.33, 99.63, 98.15, 97.48]
    xiangya_Investigator1 = [97.84, 90.00, 100.00, 100.00, 97.32]
    xiangya_Investigator2 = [96.04, 90.00, 97.71, 91.52, 97.26]
    xiangya_Investigator3 = [95.68, 83.33, 99.08, 96.15, 95.58]
    xiangya_Investigator4 = [95.32, 81.67, 99.08, 96.08, 95.15]
    xiangya_Investigator_avg = [96.22, 86.25, 98.96, 95.94, 96.33]
    xiangya_values = np.array([xiangya_ResNet18, xiangya_ResNet_little, xiangya_Investigator1,
                               xiangya_Investigator2, xiangya_Investigator3, xiangya_Investigator4,
                               xiangya_Investigator_avg])  # [7, 5]
    label_list = ['ResNet18 (T)', 'ResNet_little (Ours)', 'Investigator1', 'Investigator2', 'Investigator3',
                  'Investigator4', 'Investigator_avg']
    bar_width = 0.9  # the width of the bars
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(28, 10))  # 调整图形大小
    colors = ['#EE7D80', '#F9BE80', '#098036', '#D3D3D3', '#4B57A2', '#000000', '#FCDFC0']

    # 设置柱子的x轴位置
    index = np.arange(len(categories))
    bars = []
    for i in range(xiangya_values.shape[0]):
        bars.append(ax.bar(index * 7.2 + i * bar_width, xiangya_values[i], bar_width, label=label_list[i], color=colors[i]))

    # rects1 = ax.bar(x, huaxi_ResNet18, width, label='ResNet18 (T)')
    # rects2 = ax.bar(x + width / 2, huaxi_ResNet_little, width, label='ResNet_little (Ours)')
    # rects3 = ax.bar(x + width / 2 * 2, huaxi_Investigator1, width, label='Investigator1')
    # rects4 = ax.bar(x + width / 2 * 3, huaxi_Investigator2, width, label='Investigator2')
    # rects5 = ax.bar(x + width / 2 * 4, huaxi_Investigator3, width, label='Investigator3')
    # rects6 = ax.bar(x + width / 2 * 5, huaxi_Investigator4, width, label='Investigator4')
    # rects7 = ax.bar(x + width / 2 * 6, huaxi_Investigator_avg, width, label='Investigator_avg')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Metrics', fontsize=22, fontfamily='serif')
    ax.set_ylabel('Values (%)', fontsize=22, fontfamily='serif')
    ax.set_title(title, fontsize=26, fontfamily='serif')
    ax.set_xticks(index * 7.2  + bar_width * 3)
    ax.set_xticklabels(categories, fontfamily='serif', fontsize=22)
    plt.ylim(76, 106)
    # 设置图例字体
    legend = ax.legend(ncol=2)
    plt.setp(legend.get_texts(), fontsize=20, family='serif')  # 设置字体大小和字体家族

    # 调整x轴范围，减少左右两侧的空隙
    ax.set_xlim(-bar_width, len(categories) * 8.1 - bar_width * (7 - 1))

    # 设置纵轴数字刻度的字体大小
    ax.tick_params(axis='y', labelsize=22)

    # 在每个柱子上显示高度
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=17.,  # 设置字体大小
                        fontfamily='serif'  # 设置罗马字体
                        )

    fig.tight_layout()
    # # 保存图像为 PNG 格式，指定每英寸 1200 点分辨率
    # plt.savefig(fig_path, format='png', dpi=1200, bbox_inches='tight')
    # 保存图像为 PDF 格式
    plt.savefig(fig_path, format='pdf', bbox_inches='tight')

    # plt.show()


if __name__ == '__main__':
    barplot()
