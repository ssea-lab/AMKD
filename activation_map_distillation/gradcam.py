import os
import math
import numpy as np
import torch
import cv2
import argparse
import albumentations as A
import models.resnet as resnet
import models.resnet1 as resnet1
import models.convnext as convnext
import models.swin_transformer as swin_transformer
import matplotlib.pyplot as plt
from torchvision import transforms
from constants import OCT_DEFAULT_MEAN, OCT_DEFAULT_STD
from albumentations.pytorch.transforms import ToTensorV2
from gradcam_utils import GradCAM, show_cam_on_image, center_crop_img
from util import create_directory, seeding
from models.resnet import *
from models.convnext import *
from models.swin_transformer import *
from torch.utils.data import DataLoader
from dataset import FrameDataSetLabel
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def parse_option():
    parser = argparse.ArgumentParser('the parameters for testing the model performance')
    parser.add_argument('--cam_type', type=str, default='gradcam', choices=['gradcam', 'eigen_cam'])
    parser.add_argument('--model_s', default='resnet_little', type=str, choices=['resnet_little', 'convnext_little',
                                                                                 'resnet_lite', 'convnext_lite',
                                                                                 'swin_little', 'swin_lite'],
                        help='the student model we used.')
    parser.add_argument('--model_t', default='resnet18', type=str, choices=['resnet18', 'convnext_pico', 'swin_tiny'],
                        help='the teacher model we used.')
    parser.add_argument('--num_class', default=5, type=int, help="class num")
    parser.add_argument('--model_name', default='', type=str, help="just for placeholder")

    # parameters for data
    parser.add_argument('--crop_frame_height', type=int, default=600, help='the frame height we used.')
    parser.add_argument('--frame_height', type=int, default=512, help='the frame height we used during training.')
    parser.add_argument('--frame_width', type=int, default=1024, help='the frame width we used during training.')

    # folder num，以便用于内部数据集的交叉验证
    parser.add_argument('--fold_num', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='the fold num we used for cross validation.')

    # model parameters initialization pattern
    parser.add_argument('--pre_train', action='store_true', help="weight initialized by the weight pretrained from "
                                                                 "imageNet")
    parser.add_argument('--use_class_weight', action='store_true', help="whether do linear probing or not")
    parser.add_argument('--use_label_smoothing', action='store_true', help="whether do linear probing or not")
    parser.add_argument('--label_smoothing', type=float, default=0.1, help="whether do linear probing or not")

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

    # test_mode configuration，方便查找预训练权重的路径
    parser.add_argument('--test_pattern', type=str, default="supervised_train")

    parser.add_argument('--weight_decay', default=0.6, type=float, help="weight_decay")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
           weight decay. We use a cosine schedule for WD.""")

    parser.add_argument('--test_file', default='internal_test',
                        choices=['internal_test', 'tiff_huaxi', 'tiff_xiangya'],
                        help="the test file from different center", type=str)

    parser.add_argument('--test_kind', default='frame',
                        choices=['frame', 'volume'],
                        help="the test kind for different center", type=str)

    parser.add_argument('--batch_size', type=int, default=1, help="training batch size")
    parser.add_argument('--num_workers', type=int, default=0, help="data loader thread")
    parser.add_argument('--gpu', type=int, default=0, help='the gpu number will be used')
    parser.add_argument('--checkpoint', type=str, default='checkpoint',
                        help='the directory to load the model trained weights.')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--result_dir', default='result', type=str, help='the directory of the model testing '
                                                                         'performance')

    args = parser.parse_args()
    return args


class ResizeTransform:
    def __init__(self, im_h: int, im_w: int):
        self.height = self.feature_size(im_h)
        self.width = self.feature_size(im_w)

    @staticmethod
    def feature_size(s):
        s = math.ceil(s / 4)  # PatchEmbed
        s = math.ceil(s / 2)  # PatchMerging1
        s = math.ceil(s / 2)  # PatchMerging2
        s = math.ceil(s / 2)  # PatchMerging3
        return s

    def __call__(self, x):
        result = x.reshape(x.size(0),
                           self.height,
                           self.width,
                           x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)

        return result


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

    args.save_dir = os.path.join('checkpoint', 'distill', args.label_data, args.cam_type, args.save_postfix, str(args.fold_num))
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
    print('==> loading teacher model')
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_dict = checkpoint['state_dict']
    state = model.load_state_dict(model_dict, strict=False)
    print('loading checkpoint from {}'.format(weight_path))
    print(state)
    return model


def work_gradcam(args):
    args = path_determine(args)

    image_height = 600
    image_width = 1200
    resize_image_height = args.frame_height
    resize_image_width = args.frame_width
    num_classes = 5

    model = get_model(args, distill=True)
    model = load_weight(model, args.weight_path)
    device = torch.device('cuda:' + str(args.gpu))
    model = model.to(device)

    # 定义要去进行gradcam反向求梯度的层
    if args.model_s in ['resnet18', 'resnet_little']:
        target_layers = [model.layer4]
    elif args.model_s in ['convnext_pico', 'convnext_little']:
        target_layers = [model.stages[-1]]
    elif args.model_s in ['swin_little', 'swin_lite']:
        target_layers = [model.norm]

    device = torch.device('cuda:' + str(args.gpu))
    model = model.to(device)
    reshape_transform = None
    if args.model_s in ['swin_little', 'swin_lite']:
        reshape_transform = ResizeTransform(im_h=resize_image_height, im_w=resize_image_width)
    cam = None
    if args.cam_type == 'gradcam':
        cam = GradCAM(model=model, target_layers=target_layers, device=device, reshape_transform=reshape_transform)
    elif args.cam_type == 'eigen_cam':
        cam = EigenCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    # 定义要去进行gradcam可视化的数据集
    file_name = args.test_file
    dataset_name = file_name.split('_')[1]
    label2name_dict = {0: '宫颈炎', 1: '囊肿', 2: '外翻', 3: '高级别病变', 4: '宫颈癌'}

    # 定义帧的测试数据集
    test_set = FrameDataSetLabel(args, pattern='valid', augmentation='valid_or_test')
    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             pin_memory=True,
                             shuffle=False)

    for j, data in enumerate(test_loader):
        images, labels, _, _, img_paths = data
        if j % 20 == 0:
            print(str(j + 1), ': ', img_paths)
        target_category = labels  # 图片的标签索引
        # print(target_category)
        if args.cam_type == 'gradcam':
            grayscale_cam = cam(input_tensor=images, target_category=target_category)  # [B, H, W]
        elif args.cam_type == 'eigen_cam':
            grayscale_cam = cam(input_tensor=images, targets=None)  # [B, H, W]
        grayscale_cam = grayscale_cam[0, :]
        # 将grayscale_cam(512x1024) resize 到(600x1200)
        grayscale_cam = cv2.resize(grayscale_cam, (image_width, image_height))

        img = cv2.imread(img_paths[0])  # [H, W, 3]
        # 将图片的背景裁剪
        img = img[-image_height:, :, :]  # [H, W, 3]
        img = cv2.resize(img, (image_width, image_height))
        visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
        # plt.imshow(visualization)
        # plt.show()

        # 创建保存生成热力图的文件夹
        name = img_paths[0].split('/')[-1]
        target_category = target_category.item()
        # image_directory_path = os.path.join('visualize_image', dataset_name, str(args.fold_num),
        #                                     label2name_dict[target_category])
        # create_directory(image_directory_path)
        heatmap_directory_path = os.path.join('visualize_image_result', args.directory_path, dataset_name,
                                              str(args.fold_num), label2name_dict[target_category])
        create_directory(heatmap_directory_path)
        # 保存原始图片
        # cv2.imwrite(os.path.join(image_directory_path, name), img)
        # 保存gradcam热力图
        visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(heatmap_directory_path, name.replace('.png', '_heatmap.jpg')), visualization_bgr)
        # 保存激活图，用于计算各个蒸馏框架得到的学生模型与教师模型激活值的差异
        activation_directory_path = os.path.join('visualize_image_activation', args.directory_path, dataset_name,
                                                 str(args.fold_num), label2name_dict[target_category])
        create_directory(activation_directory_path)
        np.save(os.path.join(activation_directory_path, name.replace('.png', '_activation.npy')), grayscale_cam)


if __name__ == '__main__':
    args = parse_option()
    # 设置随机数种子
    seeding(args.seed)
    # for fold_num in [0, 1, 2, 3, 4]:
    for fold_num in [0]:
        args.fold_num = fold_num
        work_gradcam(args)
