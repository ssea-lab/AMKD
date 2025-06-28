import os
import sys
import random
import cv2
import numpy as np
import pandas as pd
import torch
import openpyxl
from torch.utils.data import Dataset, Sampler
from util import ImageProcessorUnLabel, ImageProcessorLabel
from tifffile import TiffFile
from collections import Counter, defaultdict
from copy import deepcopy


class FrameDataSetUnLabel(Dataset):
    def __init__(self, args, pattern='distilling_nolabel', augmentation='distilling_train_nolabel'):
        self.args = args
        self.imgs = self.__load_file(pattern=pattern)
        print(f"num of samples {len(self.imgs)}")
        self.img_processor = ImageProcessorUnLabel(args, augmentation=augmentation)
        print('args.distill {}'.format(args.distill))

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        img = cv2.imread(imgPath)  # [H, W, 3]
        # 在进行数据增强之前，首先把帧的背景部分裁减掉
        img = crop_background(img, self.args.crop_frame_height)  # [H, W, 3]
        img = self.img_processor(img)  # [3, H, W]

        return [img, imgPath]

    def __len__(self):
        return len(self.imgs)

    def __load_file(self, pre="dataset", pattern='distilling_nolabel'):
        if pattern in ['distilling_nolabel']:
            file_name = 'self_train_frame.xlsx'
        else:
            print('not supported pattern !')
            sys.exit(-1)

        file_path = os.path.join(pre, file_name)
        print('loading data from {}'.format(file_path))

        # 接下来处理用于自监督的帧数据和测试的帧数据，因为不用分折交叉验证
        df_frame = pd.read_excel(file_path, sheet_name='file')
        # 接下来返回文件名列表和frame类别列表
        file_names = df_frame['文件名'].tolist()
        # labels = df_frame['类别'].tolist()
        return file_names


class FrameDataSetLabel(Dataset):
    def __init__(self, args, pattern='distilling', augmentation='distilling_train'):
        self.args = args
        self.imgs, self.labels = self.__load_file(pattern=pattern, fold_num=args.fold_num)
        print(f"num of samples {len(self.imgs)}")
        self.img_processor = ImageProcessorLabel(args, augmentation=augmentation)

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        # 这里我们要根据图片的路径来获取该图片对应的激活图的路径，以及该图片对应的
        # mask路径，其中激活图的路径中存储了我们自定义的激活图，mask路径则是我们
        # 分割模型生成的分割掩码图
        imgPath_split = imgPath.split('/')
        imgPath_split_copy = deepcopy(imgPath_split)
        imgPath_split.insert(4, 'segmentation_mask')
        mask_path = '/'.join(imgPath_split)
        imgPath_split_copy.insert(4, 'activation_map')
        activation_map_path = '/'.join(imgPath_split_copy).replace('png', 'npy')
        label = self.labels[index]
        # 分别读取三张图片
        img = cv2.imread(imgPath)  # [H, W, 3]
        # mask和heatmap要按灰度图读取
        if label < 3:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # [H, W]
            # 接下来需要根据mask生成每一个位置对应的权重
            mask = mask_weight(mask, label)
        else:
            mask = np.ones_like(img, dtype=np.float32)[:, :, 0]

        # 加载.npy文件
        activation_map = np.load(activation_map_path)  # [H, W]
        # 根据activation_map位置为1.0的部分表示基底膜，囊肿，外翻的轮廓，将其对应位置的权重进行加强
        if label < 3:
            mask[activation_map == 1.0] = mask[activation_map == 1.0] * 10

        # 在进行数据增强之前，首先把帧的背景部分裁减掉
        img = crop_background(img, self.args.crop_frame_height)  # [H, W, 3]
        mask = crop_background(mask, self.args.crop_frame_height)  # [H, W]
        activation_map = crop_background(activation_map, self.args.crop_frame_height)  # [H, W]
        # 然后将图片，热力图，掩码三个部分共同输入到数据增强部分做相同的变换
        img, activation_map, mask = self.img_processor(img, activation_map, mask)

        return [img, label, activation_map, mask, imgPath]

    def __len__(self):
        return len(self.imgs)

    def __load_file(self, pre="dataset", pattern='distilling', fold_num=0):
        if pattern in ['supervised_train', 'finetuning', 'distilling']:
            file_name = 'frame_internal.xlsx'
        elif pattern in ['valid', 'test']:
            if self.args.test_file == 'internal_test':
                file_name = 'frame_internal.xlsx'
            elif self.args.test_file == 'tiff_huaxi':
                file_name = 'tiff_huaxi_frame.xlsx'
            elif self.args.test_file == 'tiff_xiangya':
                file_name = 'tiff_xiangya_frame.xlsx'
        else:
            print('not supported pattern !')
            sys.exit(-1)
        file_path = os.path.join(pre, file_name)
        print(f'loading data from {file_path}')

        l_d = {0: '第一折', 1: '第二折', 2: '第三折', 3: '第四折', 4: '第五折'}
        # 接下来需要做的是按照是否需要进行交叉验证读取的Excel表格
        if 'frame_internal.xlsx' in file_path:
            df_patient = pd.read_excel(os.path.join(pre, 'patient_internal.xlsx'), sheet_name=l_d[fold_num])
            # 接下来根据按照是训练集还是测试集读取不同的数据，训练集读取前80%的病人数据，测试集读取后20%的病人数据
            if pattern in ['supervised_train', 'finetuning', 'distilling']:
                df_patient = df_patient.iloc[:int(len(df_patient) * 0.8), :]
                print(f"训练交叉验证第{fold_num}折的数据")
            elif pattern in ['valid']:
                df_patient = df_patient.iloc[int(len(df_patient) * 0.8):, :]
                print(f"测试交叉验证第{fold_num}折的数据")
            else:
                print('not supported pattern !')
                sys.exit(-1)
            df_patient.reset_index(drop=True, inplace=True)
            # 接下来需要做的是读取frame_internal.xlsx中的数据,并根据df_patient中的病人数据筛选出对应的用来训练或测试的帧数据
            df_frame = pd.read_excel(file_path, sheet_name='file')
            df_frame = df_frame.merge(df_patient, on='病人id', how='inner')
            df_frame.reset_index(drop=True, inplace=True)
            # 接下来返回文件名列表和frame类别列表
            # 下面这一行只读取阴性图片，后面可删掉
            # df_frame = df_frame[df_frame['类别'].isin([0, 1, 2])]
            file_names = df_frame['文件名'].tolist()
            labels = df_frame['类别'].tolist()
            return file_names, labels
        else:
            # 接下来处理用于自监督的帧数据和测试的帧数据，因为不用分折交叉验证
            df_frame = pd.read_excel(file_path, sheet_name='file')
            # 接下来返回文件名列表和frame类别列表
            # df_frame = df_frame[df_frame['类别'].isin([0, 1, 2])]
            file_names = df_frame['文件名'].tolist()
            labels = df_frame['类别'].tolist()
            return file_names, labels


class TiffDataSet(Dataset):
    def __init__(self, args, pattern='supervised_train', augmentation='supervised_train', fold_num=0):
        self.args = args
        self.imgs, self.labels = self.__load_file(pattern=pattern, fold_num=fold_num)
        self.img_processor = ImageProcessorLabel(args, augmentation=augmentation)

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        label = self.labels[index]
        tif = TiffFile(imgPath)
        image_arr = tif.asarray()  # ndarray [10, H, W]
        # 由于有的tiff文件中可能会超过10帧，我们只取前10帧
        image_arr = image_arr[:10, :, :]
        volume_list = []
        for f_idx in range(image_arr.shape[0]):
            ith_frame = image_arr[f_idx]  # [H, W]
            ith_frame = np.stack((ith_frame,) * 3, axis=2)  # [H, W, 3]
            # 在进行augmentation之前，先把frame array裁成标准的图片
            ith_frame = crop_background(ith_frame, self.args.crop_frame_height)  # [H, W, 3]
            ith_frame, _, _ = self.img_processor(ith_frame, None, None)  # [3, H, W]
            volume_list.append(ith_frame)
        img = torch.stack(volume_list)  # [10, 3, H, W]
        return [img, label, imgPath]

    def __len__(self):
        return len(self.imgs)

    def __load_file(self, pre="dataset", pattern='supervised_train', fold_num=0):
        if pattern in ['supervised_train', 'finetuning', 'distilling']:
            file_name = 'tiff_internal.xlsx'
        elif pattern in ['valid', 'test']:
            if self.args.test_file == 'internal_test':
                file_name = 'tiff_internal.xlsx'
            elif self.args.test_file == 'tiff_huaxi':
                file_name = 'tiff_huaxi.xlsx'
            elif self.args.test_file == 'tiff_xiangya':
                file_name = 'tiff_xiangya.xlsx'
        else:
            print('not supported pattern !')
            sys.exit(-1)
        file_path = os.path.join(pre, file_name)

        l_d = {0: '第一折', 1: '第二折', 2: '第三折', 3: '第四折', 4: '第五折'}
        # 接下来需要做的是按照是否需要进行交叉验证读取的Excel表格
        if 'tiff_internal.xlsx' in file_path:
            df_patient = pd.read_excel(os.path.join(pre, 'patient_internal.xlsx'), sheet_name=l_d[fold_num])
            # 接下来根据按照是训练集还是测试集读取不同的数据，训练集读取前80%的病人数据，测试集读取后20%的病人数据
            if pattern in ['supervised_train', 'finetuning', 'distilling']:
                df_patient = df_patient.iloc[:int(len(df_patient) * 0.8), :]
            elif pattern in ['test']:
                df_patient = df_patient.iloc[int(len(df_patient) * 0.8):, :]
            else:
                print('not supported pattern !')
                sys.exit(-1)
            df_patient.reset_index(drop=True, inplace=True)
            # 接下来需要做的是读取frame_internal.xlsx中的数据,并根据df_patient中的病人数据筛选出对应的用来训练或测试的帧数据
            df_tiff = pd.read_excel(file_path, sheet_name='file')
            df_tiff = df_tiff.merge(df_patient, on='病人id', how='inner')
            df_tiff.reset_index(drop=True, inplace=True)
            # 接下来返回文件名列表和frame类别列表
            file_names = df_tiff['文件名'].tolist()
            labels = df_tiff['类别'].tolist()
            return file_names, labels
        else:
            # 接下来处理用于自监督的帧数据和测试的帧数据，因为不用分折交叉验证
            df_tiff = pd.read_excel(file_path, sheet_name='file')
            # 接下来返回文件名列表和frame类别列表
            file_names = df_tiff['文件名'].tolist()
            labels = df_tiff['类别'].tolist()
            return file_names, labels


def crop_background(frame_array, height):
    """
    对每一帧的背景进行裁剪
    :param frame_array: shape is [H, W, 3]
    :param height: 要裁剪的图片的图片的高度
    :return: 裁剪过后的数组
    """
    # 根据image_path判断图片来源的数据集，根据相应的数据集确定合适的裁剪策略
    # if 'huaxi' in image_path:
    #     frame_array = frame_array[:height, :, :]
    # else:
    #     frame_array = frame_array[-height:, :, :]
    frame_array = frame_array[-height:, ...]

    return frame_array


# def crop_background(frame_array):
#     """
#     对每一帧的背景进行裁剪
#     :param frame_array: shape is [H, W, 3]
#     :return: 裁剪过后的数组
#     """
#     edge = findEdge(frame_array)
#     # 根据edge对背景进行裁剪
#     frame_array = frame_array[edge:, :, :]
#     return frame_array

def findEdge(frame_array):
    # frame_array只取一个通道
    frame_array = frame_array[:, :, 0]
    edge = -1
    lineMean = np.mean(frame_array, axis=-1)
    for idx in range(10, 500):
        if np.mean(lineMean[idx - 5:idx + 5]) >= 50:
            edge = idx
            break
    if edge < 10 or edge > 500:
        edge = 500
    return edge


def mask_weight(mask, label):
    """
    根据分割掩码生成每个位置的权重，缓解激活过程中类别不均衡的问题
    :param mask:
    :param label:
    :return:
    """
    if label < 3:
        # 使用flatten方法
        mask_copy = mask.flatten()
        # 使用numpy的bincount统计频率
        counts = np.bincount(mask_copy)
        total_counts = counts.sum()
        weights = (total_counts - counts) / total_counts
        weights = weights * 1.2  # 乘以1.2扩大类别不平衡带来的差距
        pos_weights = np.zeros_like(mask, dtype=np.float32)
        # 通过循环将每个位置的权重进行赋值
        for i in range(len(counts)):
            pos_weights[mask == i] = weights[i]
        return pos_weights

    else:
        pos_weights = np.ones_like(mask, dtype=np.float32)
        return pos_weights


if __name__ == '__main__':
    # file_name = 'train.txt'
    # print(file_name.replace('.txt', '_frame.txt'))
    import random

    a = list(range(20))
    c = random.sample(a, 10)
    print(c)
