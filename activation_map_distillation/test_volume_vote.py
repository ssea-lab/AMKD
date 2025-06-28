import os
import argparse
import numpy as np
import openpyxl
import pandas as pd
import models.resnet as resnet
import models.resnet1 as resnet1
import models.convnext as convnext
import models.swin_transformer as swin_transformer
from torch.utils.data import DataLoader
from dataset import TiffDataSet
from models.resnet import *
from models.resnet1 import *
from models.convnext import *
from models.swin_transformer import *
from collections import Counter
from util import seeding, create_directory
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score
from decimal import Decimal

"""
这个文件是用来根据投票策略来预测volume类别的机制，比如一个volume中有几帧为阳性才算是volume为阳性
"""
parser = argparse.ArgumentParser('the parameters for testing the model performance')

# model configuration
parser.add_argument('--model_s', default='resnet_little', type=str, choices=['resnet_little', 'convnext_little',
                                                                             'resnet_lite', 'convnext_lite',
                                                                             'swin_little', 'swin_lite'],
                    help='the student model we used.')
parser.add_argument('--model_t', default='resnet18', type=str, choices=['resnet18', 'convnext_pico', 'swin_tiny'],
                    help='the teacher model we used.')
parser.add_argument('--num_class', default=5, help="class num", type=int)
parser.add_argument('--crop_frame_height', type=int, default=600, help='the frame height we used.')
parser.add_argument('--frame_height', type=int, default=512, help='the frame height we used.')
parser.add_argument('--frame_width', type=int, default=1024, help='the frame width we used.')
parser.add_argument('--model_name', default='', type=str, help="just for placeholder")
parser.add_argument('--cam_type', type=str, default='gradcam', choices=['gradcam', 'eigen_cam'])

# data augmentation type
parser.add_argument('--augmentation', type=str, default='valid_or_test')
# test_mode configuration，方便查找预训练权重的路径
parser.add_argument('--test_pattern', type=str, default="supervised_train")

# model parameters initialization pattern
parser.add_argument('--pre_train', action='store_true', help="weight initialized by the weight pretrained from "
                                                             "imageNet")
# 这个地方用于证明大规模无标签数据用于蒸馏时可以很好的提高模型的性能
parser.add_argument('--use_unlabel_data', action='store_true', help="whether use unlabel data for distillation")
parser.add_argument('--use_pselabel', action='store_true',
                    help="whether use psedeuo label data for unlabeled distillation")
parser.add_argument('--use_label_data', action='store_true', help="whether use label data for distillation")

# distillation
parser.add_argument('--distill', type=str, default='amkd')
# configuration for the weight of different loss function
parser.add_argument('--am_weight', type=float, default=1.0, help='weight of instance RKD loss')
parser.add_argument('--kl_weight', type=float, default=1.0, help='weight of ce loss')
parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')

# whether use the loss function
parser.add_argument('--use_am', action='store_true', help="whether use the irkd loss")
parser.add_argument('--use_kl', action='store_true', help="whether use the kl loss")
parser.add_argument('--use_ce', action='store_true', help="whether use the ce loss")

parser.add_argument('--strategy', type=str, default='vote', choices=['vote', 'continue_frames'],
                    help='the number of positive frames that we can say the volume is positive ')
parser.add_argument('--voting_number', type=int, default=1, help='the number of positive frames that we can say'
                                                                 'the volume is positive ')
parser.add_argument('--continue_pos_frames', type=int, default=3, help='the number of positive frames that we can say'
                                                                       'the volume is positive ')

parser.add_argument('--test_file', default='tiff_test',
                    choices=['internal_test', 'tiff_huaxi', 'tiff_xiangya'],
                    help="the test file from different center", type=str)

parser.add_argument('--test_kind', default='volume', choices=['volume'],
                    help="the test kind for different center", type=str)

# folder num，以便用于内部数据集的交叉验证
parser.add_argument('--fold_num', type=int, default=0, choices=[0, 1, 2, 3, 4],
                    help='the fold num we used for cross validation.')

parser.add_argument('--use_class_weight', action='store_true', help="whether do linear probing or not")
parser.add_argument('--use_label_smoothing', action='store_true', help="whether do linear probing or not")
parser.add_argument('--label_smoothing', type=float, default=0.1, help="whether do linear probing or not")

parser.add_argument('--weight_decay', default=0.6, type=float, help="weight_decay")
parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
       weight decay. We use a cosine schedule for WD.""")

parser.add_argument('--batch_size', type=int, default=1, help="training batch size")
parser.add_argument('--num_workers', type=int, default=0, help="data loader thread")
parser.add_argument('--gpu', type=int, default=0, help='the gpu number will be used')
parser.add_argument('--checkpoint', type=str, default='checkpoint',
                    help='the directory to load the model trained weights.')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--result_dir', default='result', type=str, help='the directory of the model testing '
                                                                     'performance')

args = parser.parse_args()


def path_determine(args):
    # 获取学生模型的权重存储路径
    assert args.use_am or args.use_kl or args.use_ce, '所有的损失函数不能同时为False'
    if not args.use_am:
        args.am_weight = 0
    if not args.use_kl:
        args.kl_weight = 0
    if not args.use_ce:
        args.ce_weight = 0
    args.save_postfix = 'S:{}_T:{}_distill:{}' \
                        '_am:{}_kl:{}_ce:{}'.format(args.model_s,
                                                    args.model_t, args.distill,
                                                    args.am_weight,
                                                    args.kl_weight, args.ce_weight)

    if args.use_unlabel_data and args.use_label_data:
        args.label_data = 'all_data_pselabel' if args.use_pselabel else 'all_data'
    else:
        args.label_data = 'unlabel_data' if args.use_unlabel_data else 'label_data'

    args.save_dir = os.path.join('checkpoint', 'distill', args.label_data, args.cam_type, args.save_postfix,
                                 str(args.fold_num))
    args.directory_path = os.path.join('distill', args.label_data, args.cam_type, args.save_postfix)
    args.weight_path = os.path.join(args.save_dir, 'checkpoint.pth')
    return args

def get_model(args, distill=False, teacher=False):
    model_name = args.model_name
    if distill:
        model_name = args.model_t if teacher else args.model_s
    pre_train = args.pre_train
    num_class = args.num_class
    model = None
    # 定义模型
    if model_name == 'resnet18':
        model = getattr(resnet, model_name)(pretrained=False, num_classes=num_class)
    elif model_name in ['resnet_little', 'resnet_lite']:
        model = getattr(resnet1, model_name)(num_classes=num_class)
    elif model_name in ['convnext_pico', 'convnext_little', 'convnext_lite']:
        model = getattr(convnext, model_name)(pretrained=False, num_classes=num_class)
    else:
        model = getattr(swin_transformer, model_name)(pretrained=False, num_classes=num_class)
    return model

def load_weight(model, weight_path):
    # 加载模型模型权重
    print('==> loading model weight')
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_dict = checkpoint['state_dict']
    state = model.load_state_dict(model_dict, strict=False)
    print('loading checkpoint from {}'.format(weight_path))
    print(state)
    return model


def worker_frame(args):
    args = path_determine(args)
    # 定义Tiff的测试数据集
    test_set = TiffDataSet(args, pattern='valid', augmentation='valid_or_test')
    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             pin_memory=True,
                             shuffle=False)

    # 定义模型
    model = get_model(args, distill=True)
    model = load_weight(model, args.weight_path)
    device = torch.device('cuda:' + str(args.gpu))
    model = model.to(device)

    result_dir = ''
    if args.strategy == 'vote':
        result_dir = os.path.join(args.result_dir, args.test_kind, args.test_file, args.directory_path,
                                  args.strategy, str(args.voting_number))
    elif args.strategy == 'continue_frames':
        result_dir = os.path.join(args.result_dir, args.test_kind, args.test_file, args.directory_path,
                                  args.strategy, str(args.continue_pos_frames))

    create_directory(result_dir)
    test_result_path = os.path.join(result_dir, 'test_result.xlsx')
    total_result_path = os.path.join(result_dir, 'total_result.xlsx')

    test_frame(test_loader, model, test_result_path, args.voting_number, args.continue_pos_frames, args.strategy)
    get_metrics_from_file(test_result_path, total_result_path)


def test_frame(test_loader, model, result_path, voting_number, continue_pos_frames, strategy):
    model.eval()
    device = next(model.parameters()).device

    # 定义所要输出的文件的表头
    header = ['文件名', '十帧五分类概率分布', '十帧阳性概率', '十帧五分类预测标签', '十帧二分类预测标签',
              'volume五分类预测标签', 'volume五分类概率分布', 'volume二分类预测标签', 'volume二分类概率分布',
              '真实五分类标签']
    # 创建一个excel表格用于存储预测结果
    workbook = openpyxl.Workbook()
    # 获取当前活跃的worksheet，默认就是第一个worksheet
    worksheet = workbook.active
    worksheet.title = 'result'
    # 添加结果excel文件的表头
    worksheet.append(header)
    with torch.no_grad():
        for j, data in enumerate(test_loader):
            images, labels, img_paths = data
            B, num_frames, C, H, W = images.size()
            images = images.reshape(B * num_frames, C, H, W)  # 由于batch_size设置为1，所以实际上为[num_frames, C, H, W]
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)  # [num_frames, num_class]

            # 接下来我们要把模型预测的结果和真实的标签写到结果文件中
            ten_frame_outputs = outputs.cpu().numpy()
            labels = labels.numpy()  # [1]
            ten_frame_pos_outputs = ten_frame_outputs[:, -2] + ten_frame_outputs[:, -1]  # [10]
            ten_frame_neg_outputs = ten_frame_outputs[:, 0] + ten_frame_outputs[:, 1] + ten_frame_outputs[:, 2]  # [10]
            ten_frame_five_pre_label = ten_frame_outputs.argmax(axis=1)  # [10]
            ten_frame_binary_pre_label_bool = ten_frame_pos_outputs >= ten_frame_neg_outputs
            ten_frame_binary_pre_label = np.array(ten_frame_binary_pre_label_bool, dtype=int)
            # 接下来根据voting_number有由阳性帧的数目退出volume的二分类类别
            num_pos_frames = np.sum(ten_frame_binary_pre_label)
            # 根据连续几帧为阳的策略判断volume的二分类类别
            ten_frame_binary_pre_label_continue_str = ''.join(list(map(str, ten_frame_binary_pre_label.tolist())))
            pos_pattern = '1' * continue_pos_frames
            if (strategy == 'vote' and num_pos_frames >= voting_number) or (
                    strategy == 'continue_frames' and pos_pattern in ten_frame_binary_pre_label_continue_str):
                volume_binary_label = 1
                # 二分类阳性概率的计算，我们使用阳性帧概率的平均值做为volume阳性的概率
                volume_binary_prob = np.sum(ten_frame_pos_outputs[ten_frame_binary_pre_label_bool]) / num_pos_frames
            else:
                volume_binary_label = 0
                volume_binary_prob = np.sum(ten_frame_neg_outputs[~ten_frame_binary_pre_label_bool]) / (len(
                    ten_frame_neg_outputs) - num_pos_frames)

            # 统计10帧中五个类别中各个类别的帧数，由此推断得到volume的五分类类别
            frame_five_class_counter = Counter(ten_frame_five_pre_label)
            # 根据二分类的volume的label选择使用计算五分类类别的类别数目
            if volume_binary_label == 1:
                if frame_five_class_counter[4] > 0:
                    volume_five_label = 4
                else:
                    volume_five_label = 3
            else:
                # 这个时候我们便要从炎症，囊肿，外翻三个类别中进行选择
                if frame_five_class_counter[2] >= 2:
                    volume_five_label = 2

                elif frame_five_class_counter[1] >= 2:
                    volume_five_label = 1
                else:
                    volume_five_label = 0
            # 接下来计算volume 五分类label对应的类别概率
            class_frames = ten_frame_outputs[ten_frame_five_pre_label == volume_five_label]
            volume_five_prob = np.sum(class_frames[:, volume_five_label]) / class_frames.shape[0]

            # 接下来需要将内容输出到文件中以便保存中间的结果，以及计算各种指标
            img_path = img_paths[0]
            content = [img_path, np.array_str(ten_frame_outputs), np.array_str(ten_frame_pos_outputs),
                       np.array_str(ten_frame_five_pre_label), np.array_str(ten_frame_binary_pre_label),
                       volume_five_label, volume_five_prob, volume_binary_label,
                       volume_binary_prob, labels[0]]
            worksheet.append(content)

        workbook.save(filename=result_path)


def get_metrics_from_file(test_result_file, total_result_file):
    # 所要计算的指标包括五分类准确率(five class accuracy)，五分类的mac_croF1, 二分类准确率(binary class accuracy),
    # 灵敏度(sensitivity), 特异性(specificity), PPV, NPV, AUC
    # test_result_file 输出结果的类别格式
    # image_path, 5分类的预测概率, 5分类的预测标签, 5分类的真实标签
    binary_class_pre = []
    binary_class_labels = []
    binary_class_probabilities = []
    df = pd.read_excel(io=test_result_file, sheet_name='result',
                       usecols=['文件名', 'volume二分类预测标签', 'volume二分类概率分布', '真实二分类标签'])

    for idx, row in df.iterrows():
        img_path = row['文件名']
        binary_pre = row['volume二分类预测标签']
        if binary_pre == 1:
            pos_p = float(row['volume二分类概率分布'])
        else:
            pos_p = 1 - float(row['volume二分类概率分布'])
        five_label = row['真实五分类标签']
        binary_label = 0 if five_label < 3 else 1

        binary_class_pre.append(binary_pre)
        binary_class_labels.append(binary_label)
        binary_class_probabilities.append(pos_p)

    binary_class_pre = np.array(binary_class_pre)
    binary_class_labels = np.array(binary_class_labels)
    binary_class_probabilities = np.array(binary_class_probabilities)

    # 利用sklearn的接口计算得到五分类的准确率和二分类的准确率
    # 接下来利用二分类的混淆矩阵计算各种指标，包括灵敏度，特异性，PPV, NPV
    # metrics = [binary_accracy, sensitivity, specifity, ppv, npv, binary_confusion_matrix]
    metrics = cal_bin_metrics(y_true=binary_class_labels, y_pred=binary_class_pre)
    binary_accracy, sensitivity, specifity, ppv, npv, binary_confusion_matrix = metrics[0], metrics[1], metrics[2], \
        metrics[3], metrics[4], metrics[5]

    # 接下来计算二分类的auc的值
    auc = roc_auc_score(y_true=binary_class_labels, y_score=binary_class_probabilities)
    # 定义所要输出的文件的表头
    header = ['二分类准确率', '灵敏度', '特异性', 'PPV', 'NPV', 'AUC', '二分类混淆矩阵']
    # 创建一个excel表格用于存储预测结果
    workbook = openpyxl.Workbook()
    # 获取当前活跃的worksheet，默认就是第一个worksheet
    worksheet = workbook.active
    worksheet.title = 'result'
    # 添加结果excel文件的表头
    worksheet.append(header)
    metrics = [binary_accracy, sensitivity, specifity, ppv, npv, auc,
               np.array_str(binary_confusion_matrix)]
    worksheet.append(metrics)

    workbook.save(filename=total_result_file)


def cal_bin_metrics(y_true, y_pred):
    """
    the function is used to calculate the corresponding metrics for binary classification,
    including 二分类准确率，灵敏度，特异性，PPV, NPV, AUC
    :param y_true:
    :param y_pred:
    :return:
    """
    # 首先判断一下y_true, y_pred的维度，再进行相应的操作
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    binary_accracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    # 利用sklearn的接口计算得到混淆矩阵
    binary_confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    # print(y_true)
    # print(y_pred)
    # print(binary_confusion_matrix)
    # 接下来利用二分类的混淆矩阵计算各种指标，包括灵敏度，特异性，PPV, NPV
    tp = binary_confusion_matrix[1][1]
    fn = binary_confusion_matrix[1][0]
    fp = binary_confusion_matrix[0][1]
    tn = binary_confusion_matrix[0][0]
    sensitivity = tp / (tp + fn)
    specifity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    metrics = [binary_accracy, sensitivity, specifity, ppv, npv, binary_confusion_matrix]
    return metrics


def norm_value(number, percent=True):
    if percent:
        number = Decimal(number).quantize(Decimal('0.0001'), rounding='ROUND_HALF_UP') * 100
    else:
        number = Decimal(number).quantize(Decimal('0.0001'), rounding='ROUND_HALF_UP')
    return number


if __name__ == '__main__':
    args = parser.parse_args()
    # 设置随机数种子
    seeding(args.seed)
    worker_frame(args)
