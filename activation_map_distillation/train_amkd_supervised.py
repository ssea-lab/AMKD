"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import timm
import models.resnet as resnet
import models.resnet1 as resnet1
import models.convnext as convnext
import models.swin_transformer as swin_transformer
import shutil

from util import save_checkpoint, cosine_scheduler, seeding, create_directory
from dataset import FrameDataSetLabel
from torch.utils.data import DataLoader
# from test import get_model
from torch.optim import AdamW
from models.resnet import *
from models.resnet1 import *
from models.convnext import *
from models.swin_transformer import *
from models.resnet import ResNet
from models.resnet1 import ResNet as ResNet1
from distiller_zoo import DistillKL
from engine_distill import train_distill_label, train_distill_label_for_eigen_cam


def get_parse():
    parser = argparse.ArgumentParser('argument for AMKD distillation training')
    parser.add_argument('--model_s', default='resnet_little', type=str, choices=['resnet_little', 'convnext_little',
                                                                                 'resnet_lite', 'convnext_lite',
                                                                                 'swin_little', 'swin_lite'],
                        help='the student model we used.')
    parser.add_argument('--model_t', default='resnet18', type=str, choices=['resnet18', 'convnext_pico', 'swin_tiny'],
                        help='the teacher model we used.')
    parser.add_argument('--num_class', default=5, type=int, help="class num")
    parser.add_argument('--model_name', default='', type=str, help="just for placeholder")
    parser.add_argument('--cam_type', type=str, default='gradcam', choices=['gradcam', 'eigen_cam'])

    # parameters for data
    parser.add_argument('--crop_frame_height', type=int, default=600, help='the frame height we used.')
    parser.add_argument('--frame_height', type=int, default=512, help='the frame height we used during training.')
    parser.add_argument('--frame_width', type=int, default=1024, help='the frame width we used during training.')
    # folder num，以便用于内部数据集的交叉验证
    parser.add_argument('--fold_num', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='the fold num we used for cross validation.')
    parser.add_argument('--pre_train', action='store_true', help="weight initialized by the weight pretrained from ")
    parser.add_argument('--use_class_weight', action='store_true', help="whether do linear probing or not")
    parser.add_argument('--use_label_smoothing', action='store_true', help="whether do linear probing or not")
    parser.add_argument('--label_smoothing', type=float, default=0.15, help="whether do linear probing or not")
    # 这两个参数用于确定教师权重的保存路径
    parser.add_argument('--align_weight', type=float, default=1.0, help="激活图对齐损失的权重")
    parser.add_argument('--cls_weight', type=float, default=1.0, help="分类损失的权重")

    # train_mode configuration
    parser.add_argument('--train_mode', type=str, default="distilling")
    parser.add_argument('--train_pattern', type=str, default="distilling", choices=['distilling'])

    # coresponded data augmentation type for various train pattern
    parser.add_argument('--augmentation', type=str, default='distilling_train',
                        choices=['distilling_train', 'val_or_test'],
                        help="augmentation type for different training pattern")

    # distillation
    parser.add_argument('--distill', type=str, default='amkd')
    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # configuration for the weight of different loss function
    parser.add_argument('--am_weight', type=float, default=1.0, help='weight of instance RKD loss')
    parser.add_argument('--kl_weight', type=float, default=1.0, help='weight of ce loss')
    parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')

    # whether use the loss function
    parser.add_argument('--use_am', action='store_true', help="whether use the irkd loss")
    parser.add_argument('--use_kl', action='store_true', help="whether use the kl loss")
    parser.add_argument('--use_ce', action='store_true', help="whether use the ce loss")
    # training configuration
    parser.add_argument('--epochs', type=int, default=40, help="training epoch")
    parser.add_argument('--start_epoch', type=int, default=0, help="start epoch")
    parser.add_argument('--batch_size', type=int, default=32, help="training batch size")
    parser.add_argument('--num_workers', type=int, default=2, help="data loader thread")

    parser.add_argument('--gpu', type=int, default=0, help='the gpu number will be used')
    parser.add_argument('--checkpoint', type=str, default='checkpoint',
                        help='the directory to save the model weights.')
    parser.add_argument('--seed', default=42, type=int)

    # optimizer
    parser.add_argument('--optimizer', default='AdamW', type=str, help='optimizer')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--warmup_lr', type=float, default=5e-7,
                        help='warmup learning rate (default: 5e-7)')
    parser.add_argument('--min_lr', type=float, default=5e-7,
                        help='lower lr bound for cyclic schedulers that hit 0 (5e-7)')
    parser.add_argument('--warmup_epochs', type=int, default=7,
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--weight_decay', default=0.2, type=float, help="weight_decay")
    parser.add_argument('--weight_decay_end', type=float, default=0.1, help="""Final value of the
           weight decay. We use a cosine schedule for WD.""")

    args = parser.parse_args()
    return args


def path_determine(args):
    param_pattern = 'imagenet_pretrain' if args.pre_train else 'random_initial'
    args.directory_path = os.path.join('supervised', args.model_t, param_pattern,
                                       str(args.fold_num))
    args.teacher_weight_path = os.path.join('checkpoint', args.directory_path, 'checkpoint.pth')

    # 定义学生模型的存储路径以及存储的文件名
    # 首先根据使不使用对应的损失函数确定对应的损失的权重
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
    args.save_dir = os.path.join('checkpoint', 'distill', args.cam_type, args.save_postfix, str(args.fold_num))
    create_directory(args.save_dir)
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
    # 加载教师模型权重
    print('==> loading teacher model')
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_dict = checkpoint['state_dict']
    state = model.load_state_dict(model_dict, strict=False)
    print('loading checkpoint from {}'.format(weight_path))
    print(state)
    return model


def main_label_distill(args):
    args = path_determine(args)
    # 定义有标签蒸馏所用的数据集
    train_set = FrameDataSetLabel(args, pattern=args.train_pattern, augmentation=args.augmentation)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True)

    # 定义教师网络
    model_t = get_model(args, distill=True, teacher=True)
    # 加载教师网络权重，由于没有权重，先不加载
    # model_t = load_weight(model_t, args.teacher_weight_path)

    # 定义学生网络
    model_s = get_model(args, distill=True, teacher=False)
    model_t.eval()
    model_s.eval()
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    device = torch.device('cuda:' + str(args.gpu))
    # 定义损失函数
    class_weight = torch.FloatTensor([0.6, 0.8, 0.8, 0.8, 0.87])
    if args.use_class_weight:
        criterion_ce = nn.CrossEntropyLoss(weight=class_weight)
        if args.use_label_smoothing:
            criterion_ce = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)
    else:
        criterion_ce = nn.CrossEntropyLoss()
        if args.use_label_smoothing:
            criterion_ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    criterion_kl = DistillKL(args.kd_T)
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_ce)  # classification loss
    criterion_list.append(criterion_kl)  # KL divergence loss, original knowledge distillation

    # optimizer
    # 定义优化器
    optimizer = AdamW(trainable_list.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    # 将模型列表和损失函数移动到gpu上去
    module_list = module_list.to(device)
    criterion_list = criterion_list.to(device)

    # using the cosine learning rate scheduler
    num_ite_per_epoch = len(train_set) // args.batch_size
    lr_schedule_values = cosine_scheduler(args.lr, args.min_lr, args.epochs, num_ite_per_epoch, args.warmup_epochs)

    # using the cosine weight decay scheduler
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, num_ite_per_epoch)

    # routine
    for epoch in range(args.epochs):
        if args.cam_type == 'gradcam':
            train_loss = train_distill_label(train_loader, module_list, criterion_list, optimizer, args,
                                             num_ite_per_epoch, lr_schedule_values,
                                             wd_schedule_values, epoch)
        elif args.cam_type == 'eigen_cam':
            train_loss = train_distill_label_for_eigen_cam(train_loader, module_list, criterion_list, optimizer, args,
                                                           num_ite_per_epoch, lr_schedule_values,
                                                           wd_schedule_values, epoch)

        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            state = {
                'epoch': epoch + 1,
                'state_dict': model_s.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(state, checkpoint_dir=args.save_dir, is_best=False)


if __name__ == '__main__':
    args = get_parse()
    seeding(args.seed)
    for fold_num in [2]:
    # for fold_num in [3, 4]:
        args.fold_num = fold_num
        main_label_distill(args)
