import numpy as np


def calculate_f1(confusion_matrix):
    # 提取混淆矩阵中的值
    TP = confusion_matrix[1, 1]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]
    TN = confusion_matrix[0, 0]

    # 计算精确率和召回率
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0

    # 计算F1值
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return f1


def calculate_g_mean(confusion_matrix):
    # 提取混淆矩阵中的值
    TP = confusion_matrix[1, 1]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]
    TN = confusion_matrix[0, 0]

    # 计算Sensitivity和Specificity
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    # 计算G-Mean
    g_mean = np.sqrt(sensitivity * specificity)

    return g_mean


import numpy as np

import numpy as np
import random



def cal_g_means_interval(confusion_matrix):
    # Bootstrap过程
    n_bootstraps = 1000
    bootstrap_G_means = []

    # 重复抽样并计算G-mean
    for _ in range(n_bootstraps):
        bootstrap_sample = np.array([random.choices(confusion_matrix[i], k=2) for i in range(2)]).reshape(2, 2)
        bootstrap_G_mean = calculate_g_mean(bootstrap_sample)
        bootstrap_G_means.append(bootstrap_G_mean)

    # 计算置信区间
    bootstrap_G_means = np.array(bootstrap_G_means)
    CI_lower = np.percentile(bootstrap_G_means, 2.5)
    CI_upper = np.percentile(bootstrap_G_means, 97.5)
    return CI_lower, CI_upper


if __name__ == '__main__':
    # 示例混淆矩阵
    conf_matrix_list = []
    model_t = np.array([[665, 14], [8, 73]])
    model_s = np.array([[663, 16], [10, 71]])
    inv1 = np.array([[678, 1], [10, 71]])
    inv2 = np.array([[670, 0], [12, 69]])
    inv3 = np.array([[664, 15], [12, 69]])
    inv4 = np.array([[660, 19], [12, 69]])
    conf_matrix_list.append(model_t)
    conf_matrix_list.append(model_s)
    conf_matrix_list.append(inv1)
    conf_matrix_list.append(inv2)
    conf_matrix_list.append(inv3)
    conf_matrix_list.append(inv4)
    for conf_matrix in conf_matrix_list:
        # f1_score = calculate_g_mean(conf_matrix)
        # print(f1_score)
        # 计算G-mean和95%置信区间
        num_samples = 1000  # Bootstrap采样数量
        alpha = 0.05  # 置信水平95%
        data = np.array([0] * conf_matrix[0, 0] +
                        [1] * conf_matrix[0, 1] +
                        [2] * conf_matrix[1, 0] +
                        [3] * conf_matrix[1, 1])
        lower_bound, upper_bound = cal_g_means_interval(conf_matrix)
        print(lower_bound, upper_bound)

    print('-' * 10)

    conf_matrix_list = []
    model_t = np.array([[216, 2], [5, 55]])
    model_s = np.array([[217, 1], [7, 53]])
    inv1 = np.array([[218, 0], [6, 54]])
    inv2 = np.array([[213, 5], [6, 54]])
    inv3 = np.array([[216, 2], [10, 50]])
    inv4 = np.array([[216, 2], [11, 49]])
    conf_matrix_list.append(model_t)
    conf_matrix_list.append(model_s)
    conf_matrix_list.append(inv1)
    conf_matrix_list.append(inv2)
    conf_matrix_list.append(inv3)
    conf_matrix_list.append(inv4)
    for conf_matrix in conf_matrix_list:
        # f1_score = calculate_g_mean(conf_matrix)
        # print(f1_score)
        lower_bound, upper_bound = cal_g_means_interval(conf_matrix)
        print(lower_bound, upper_bound)
