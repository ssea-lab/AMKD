import time
import torch
import random
import math
import numpy as np
import sys
import torch.nn.functional as F
from util import DataUpdater, Matrics
from timm.models.convnext import DropPath
from models.resnet import ResNet
from models.resnet1 import ResNet as ResNet1
from models.convnext import ConvNeXt
from models.swin_transformer import SwinTransformer
from models.swin_transformer import DropPath as SwinDropPath


def eval_bn(model):
    # 遍历模型中的所有模块，并将批量正则化层设置为评估模式
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            # print(name, '设置为eval()状态')
    return model


def train_bn(model):
    # 遍历模型中的所有模块，并将批量正则化层设置为训练模式
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train()
            # print(name, '设置为train()状态')

    return model


def deactivate_drop_path(model):
    # 遍历模型中的所有模块，并将批量正则化层设置为训练模式
    for name, module in model.named_modules():
        if isinstance(module, (DropPath, SwinDropPath)):
            module.drop_prob = 0.0
            # print(name, 'drop_prob设置为0.0')
    return model


def activate_drop_path(model):
    # 遍历模型中的所有模块，并将批量正则化层设置为训练模式
    drop_path_rate = 0.1
    depths = model.depths
    dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths))]
    index = 0
    for name, module in model.named_modules():
        if isinstance(module, (DropPath, SwinDropPath)):
            module.drop_prob = dp_rates[index]
            index += 1
            # print(name, 'drop_prob设置为{}'.format( dp_rates[index]))
    return model


def get_2d_projection(activation_batch):
    # activation_batch: [B, C, H, W]
    # TBD: use pytorch batch svd implementation
    # activation_batch[torch.isnan(activation_batch)] = 0
    corrected_activation_batch = torch.where(torch.isnan(activation_batch), torch.zeros_like(activation_batch),
                                             activation_batch)
    B, C, H, W = corrected_activation_batch.size()
    reshaped_activations = corrected_activation_batch.reshape(B, C, -1).transpose(1, 2)  # [B, HW, C]
    # Centering before the SVD seems to be important here,
    # Otherwise the image returned is negative
    reshaped_activations = reshaped_activations - reshaped_activations.mean(axis=1, keepdims=True)  # [B, HW, C]
    U, S, VT = torch.linalg.svd(reshaped_activations, full_matrices=True,  driver='gesvd')  # [B, HW, HW], [B, HW, C], [B, C, C]
    projection = torch.matmul(reshaped_activations, VT[:, 0, :][:, :, None]).squeeze(2)  # [B, HW]
    projection = projection.reshape(B, H, W)
    projection = projection.to(torch.float32)
    return projection


# 初始化用于存储梯度的列表
grads_t = None
grads_s = None


# 定义后向钩子函数
def backward_hook_t(module, grad_input, grad_output):
    global grads_t
    grads_t = grad_output[0]
    # print(grads.size())


def backward_hook_s(module, grad_input, grad_output):
    global grads_s
    grads_s = grad_output[0]
    # print(grads.size())


def feature_size(s):
    s = math.ceil(s / 4)  # PatchEmbed
    s = math.ceil(s / 2)  # PatchMerging1
    s = math.ceil(s / 2)  # PatchMerging2
    s = math.ceil(s / 2)  # PatchMerging3
    return s


def upsample(cam, target_height, target_width):
    """
    将模型生成的cam通过双线性插值的方法进行32倍上采样
    :param cam: [B, H, W]
    :param target_height: 自定义激活图的高度
    :param target_width: 自定义激活图的宽度
    :return:
    """
    # 添加一个通道维度，使形状变为 [B, C, H, W]
    cam = cam.unsqueeze(1)  # 现在形状为 [10, 1, 64, 64]

    # 使用插值进行下采样
    upsampled_cam = F.interpolate(cam, size=(target_height, target_width), mode='bilinear',
                                  align_corners=False)
    upsampled_cam = upsampled_cam.squeeze(1)

    return upsampled_cam


def train_distill_unlabel(train_loader, module_list, criterion_kl, optimizer, args,
                          num_ite_per_epoch, lr_schedule_values,
                          wd_schedule_values, ith_epoch):
    """
    这个函数使用对蒸馏训练的一个epoch进行
    One epoch distillation
    """
    # 设置学生模型为训练模式
    module_list[0].train()
    # 设置教师模型为eval()模式
    module_list[1].eval()

    model_s = module_list[0]
    model_t = module_list[1]

    # 将学生模型和教师模型的最后一个阶段的模块引入钩子函数
    for model in [model_s]:
        if isinstance(model, (ResNet, ResNet1)):
            hook = model.layer4.register_backward_hook(backward_hook_s)
        elif isinstance(model, ConvNeXt):
            hook = model.stages[-1].register_backward_hook(backward_hook_s)
        elif isinstance(model, SwinTransformer):
            hook = model.norm.register_backward_hook(backward_hook_s)

    for model in [model_t]:
        if isinstance(model, (ResNet, ResNet1)):
            hook = model.layer4.register_backward_hook(backward_hook_t)
        elif isinstance(model, ConvNeXt):
            hook = model.stages[-1].register_backward_hook(backward_hook_t)
        elif isinstance(model, SwinTransformer):
            hook = model.norm.register_backward_hook(backward_hook_t)

    batch_time = DataUpdater()
    losses = DataUpdater()
    kl_losses = DataUpdater()
    align_losses = DataUpdater()
    metrics = Matrics()
    tm = time.time()
    device = next(model_s.parameters()).device

    for j, data in enumerate(train_loader):
        images, _ = data
        images = images.to(device, non_blocking=True)
        batch_size = images.shape[0]
        image_height, image_width = images.size(2), images.size(3)
        # 首先利用反向传播求出教师模型的Activation map
        stage4_feature_t = model_t.forward_features(images)  # [B, C_t, H, W]
        logits_t = model_t.forward_head(stage4_feature_t)  # [B, num_class]
        class_t = torch.argmax(logits_t, dim=1)  # [B]

        # 接下来根据logits_t的预测类别反向传播求stage4_feature_t的梯度
        # 计算每个样本的预测类别对应的梯度
        # 将教师模型的梯度清零
        model_t.zero_grad()
        target = torch.zeros_like(logits_t)
        for i in range(batch_size):
            target[i, class_t[i]] = 1
        logits_t.backward(gradient=target, retain_graph=True)

        # 这个地方要做判断是否为swin-transformer，从而对梯度和特征图进行reshape操作
        if isinstance(model_t, SwinTransformer):
            h, w = feature_size(image_height), feature_size(image_width)
            # 将特征图进行reshape操作
            stage4_feature_t = stage4_feature_t.reshape(stage4_feature_t.size(0), h, w, stage4_feature_t.size(2))
            stage4_feature_t = stage4_feature_t.permute(0, 3, 1, 2)  # [batch_size, H, W, C] -> [batch, C, H, W]
            # 将梯度进行reshape操作
            global grads_t
            grads_t = grads_t.reshape(grads_t.size(0), h, w, grads_t.size(2))
            grads_t = grads_t.permute(0, 3, 1, 2)  # [batch_size, H, W, C] -> [batch, C, H, W]

        grads_detach_t = grads_t.detach()  # [B, C, H, W]
        weights_t = torch.mean(grads_detach_t, [2, 3], keepdim=True)  # [B, C, 1, 1]
        cam_t = torch.sum(weights_t * stage4_feature_t, dim=1)  # [B, H, W]
        cam_t = torch.relu(cam_t)
        # 应用最小最大归一化
        cam_t = cam_t - cam_t.reshape(batch_size, -1).min(dim=1, keepdim=True)[0].reshape(batch_size, 1, 1)
        cam_t = cam_t / (cam_t.reshape(batch_size, -1).max(dim=1, keepdim=True)[0].reshape(batch_size, 1, 1) + 1e-7)

        # ================下面求学生模型的梯度======================

        if isinstance(model_s, (ResNet, ResNet1)):
            # 将模型的Batch Normalization设置为eval()，保证与推理过程中一致
            model_s = eval_bn(model_s)
        elif isinstance(model_s, (ConvNeXt, SwinTransformer)):
            model_s = deactivate_drop_path(model_s)

        stage4_feature_s = model_s.forward_features(images)  # [B, C_s, H, W]
        logits_s = model_s.forward_head(stage4_feature_s)  # [B, num_class]
        # 接下来根据logits_t的预测类别反向传播求stage4_feature_s的梯度
        # 计算每个样本的预测类别对应的梯度
        # 将学生模型的梯度清零，由于优化器只封装了学生模型的参数，所以需要将所有参数清零
        optimizer.zero_grad()
        logits_s.backward(gradient=target, retain_graph=True)
        # 这个地方要做判断是否为swin-transformer，从而对梯度和特征图进行reshape操作
        if isinstance(model_s, SwinTransformer):
            h, w = feature_size(image_height), feature_size(image_width)
            # 将特征图进行reshape操作
            stage4_feature_s = stage4_feature_s.reshape(stage4_feature_s.size(0), h, w, stage4_feature_s.size(2))
            stage4_feature_s = stage4_feature_s.permute(0, 3, 1, 2)  # [batch_size, H, W, C] -> [batch, C, H, W]
            # 将梯度进行reshape操作
            global grads_s
            grads_s = grads_s.reshape(grads_s.size(0), h, w, grads_s.size(2))
            grads_s = grads_s.permute(0, 3, 1, 2)  # [batch_size, H, W, C] -> [batch, C, H, W]

        grads_detach_s = grads_s.detach()  # [B, C, H, W]
        weights_s = torch.mean(grads_detach_s, [2, 3], keepdim=True)  # [B, C, 1, 1]
        cam_s = torch.sum(weights_s * stage4_feature_s, dim=1)  # [B, H, W]
        cam_s = torch.relu(cam_s)
        # 应用最小最大归一化
        cam_s = cam_s - cam_s.reshape(batch_size, -1).min(dim=1, keepdim=True)[0].reshape(batch_size, 1, 1)
        cam_s = cam_s / (cam_s.reshape(batch_size, -1).max(dim=1, keepdim=True)[0].reshape(batch_size, 1, 1) + 1e-7)
        # 计算激活图对齐损失
        loss_act_align = F.smooth_l1_loss(cam_s, cam_t.detach())

        # 将模型的BN层设置为train()状态，与正常训练保持一致
        if isinstance(model_s, (ResNet, ResNet1)):
            model_s = train_bn(model_s)
        elif isinstance(model_s, ConvNeXt):
            model_s = activate_drop_path(model_s)

        outputs = model_s(images)  # batch_size, num_class
        # 计算kl蒸馏损失
        loss_kl = criterion_kl(outputs, logits_t.detach())
        loss = args.am_weight * loss_act_align + args.kl_weight * loss_kl

        # 根据cosine scheduler 修改optimizer的learning rate 和 weight decay
        global_step = ith_epoch * num_ite_per_epoch + j
        for _, param_group in enumerate(optimizer.param_groups):
            if lr_schedule_values is not None:
                param_group["lr"] = lr_schedule_values[global_step]
            if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_schedule_values[global_step]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), batch_size)
        align_losses.update(args.am_weight * loss_act_align.item(), batch_size)
        kl_losses.update(args.kl_weight * loss_kl.item(), batch_size)
        batch_time.update(time.time() - tm)
        tm = time.time()
        if j % 80 == 0:
            print(
                f'Train Epoch: {ith_epoch + 1} [{(j + 1) * batch_size}/{len(train_loader.dataset)} ({100. * ((j + 1) * batch_size) / len(train_loader.dataset) :.2f}%)]\
                                 Loss: {losses.val:.4f} (Avg:{losses.avg:.4f})\
                                 Align_Loss: {align_losses.val:.4f} (Avg:{align_losses.avg:.4f})\
                                 KL_Loss: {kl_losses.val:.4f} (Avg:{kl_losses.avg:.4f})\
                                 Time: {batch_time.val:.3f} (Avg:{batch_time.avg:.3f})')

    print(f'the {ith_epoch + 1} epoch training time {batch_time.sum:.3f}\n')
    return losses.avg


def train_distill_label(train_loader, module_list, criterion_list, optimizer, args,
                        num_ite_per_epoch, lr_schedule_values,
                        wd_schedule_values, ith_epoch):
    """
    这个函数使用对蒸馏训练的一个epoch进行
    One epoch distillation
    """
    # 设置学生模型为训练模式
    module_list[0].train()
    # 设置教师模型为eval()模式
    module_list[1].eval()

    model_s = module_list[0]
    model_t = module_list[1]

    # 将学生模型和教师模型的最后一个阶段的模块引入钩子函数
    # 将学生模型和教师模型的最后一个阶段的模块引入钩子函数
    for model in [model_s]:
        if isinstance(model, (ResNet, ResNet1)):
            hook = model.layer4.register_backward_hook(backward_hook_s)
        elif isinstance(model, ConvNeXt):
            hook = model.stages[-1].register_backward_hook(backward_hook_s)
        elif isinstance(model, SwinTransformer):
            hook = model.norm.register_backward_hook(backward_hook_s)

    for model in [model_t]:
        if isinstance(model, (ResNet, ResNet1)):
            hook = model.layer4.register_backward_hook(backward_hook_t)
        elif isinstance(model, ConvNeXt):
            hook = model.stages[-1].register_backward_hook(backward_hook_t)
        elif isinstance(model, SwinTransformer):
            hook = model.norm.register_backward_hook(backward_hook_t)

    criterion_ce = criterion_list[0]
    criterion_kl = criterion_list[1]

    batch_time = DataUpdater()
    losses = DataUpdater()
    ce_losses = DataUpdater()
    kl_losses = DataUpdater()
    align_losses = DataUpdater()
    top1 = DataUpdater()
    metrics = Matrics()
    tm = time.time()
    device = next(model_s.parameters()).device

    for j, data in enumerate(train_loader):
        # 记录开始时间
        start_time = time.time()
        images, labels, _, _, image_paths = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.shape[0]
        image_height, image_width = images.size(2), images.size(3)
        # 首先利用反向传播求出教师模型的Activation map
        stage4_feature_t = model_t.forward_features(images)  # [B, C_t, H, W]
        logits_t = model_t.forward_head(stage4_feature_t)  # [B, num_class]

        # 接下来根据logits_t的预测类别反向传播求stage4_feature_t的梯度
        # 计算每个样本的标签对应的梯度
        # 将教师模型的梯度清零
        model_t.zero_grad()
        target = torch.zeros_like(logits_t)
        for i in range(batch_size):
            target[i, int(labels[i])] = 1
        logits_t.backward(gradient=target, retain_graph=True)

        # 这个地方要做判断是否为swin-transformer，从而对梯度和特征图进行reshape操作
        if isinstance(model_t, SwinTransformer):
            h, w = feature_size(image_height), feature_size(image_width)
            # 将特征图进行reshape操作
            stage4_feature_t = stage4_feature_t.reshape(stage4_feature_t.size(0), h, w, stage4_feature_t.size(2))
            stage4_feature_t = stage4_feature_t.permute(0, 3, 1, 2)  # [batch_size, H, W, C] -> [batch, C, H, W]
            # 将梯度进行reshape操作
            global grads_t
            grads_t = grads_t.reshape(grads_t.size(0), h, w, grads_t.size(2))
            grads_t = grads_t.permute(0, 3, 1, 2)  # [batch_size, H, W, C] -> [batch, C, H, W]

        grads_detach_t = grads_t.detach()  # [B, C, H, W]
        weights_t = torch.mean(grads_detach_t, [2, 3], keepdim=True)  # [B, C, 1, 1]
        cam_t = torch.sum(weights_t * stage4_feature_t, dim=1)  # [B, H, W]
        cam_t = torch.relu(cam_t)
        # 应用最小最大归一化
        cam_t = cam_t - cam_t.reshape(batch_size, -1).min(dim=1, keepdim=True)[0].reshape(batch_size, 1, 1)
        cam_t = cam_t / (cam_t.reshape(batch_size, -1).max(dim=1, keepdim=True)[0].reshape(batch_size, 1, 1) + 1e-7)

        # ================下面求学生模型的梯度======================

        if isinstance(model_s, (ResNet, ResNet1)):
            # 将模型的Batch Normalization设置为eval()，保证与推理过程中一致
            model_s = eval_bn(model_s)
        elif isinstance(model_s, (ConvNeXt, SwinTransformer)):
            model_s = deactivate_drop_path(model_s)

        stage4_feature_s = model_s.forward_features(images)  # [B, C_s, H, W]
        logits_s = model_s.forward_head(stage4_feature_s)  # [B, num_class]
        # 接下来根据标签类别反向传播求stage4_feature_s的梯度
        # 计算每个样本的类别对应的梯度
        # 将学生模型的梯度清零，由于优化器只封装了学生模型的参数，所以需要将所有参数清零
        optimizer.zero_grad()
        logits_s.backward(gradient=target, retain_graph=True)

        # 这个地方要做判断是否为swin-transformer，从而对梯度和特征图进行reshape操作
        if isinstance(model_s, SwinTransformer):
            h, w = feature_size(image_height), feature_size(image_width)
            # 将特征图进行reshape操作
            stage4_feature_s = stage4_feature_s.reshape(stage4_feature_s.size(0), h, w, stage4_feature_s.size(2))
            stage4_feature_s = stage4_feature_s.permute(0, 3, 1, 2)  # [batch_size, H, W, C] -> [batch, C, H, W]
            # 将梯度进行reshape操作
            global grads_s
            grads_s = grads_s.reshape(grads_s.size(0), h, w, grads_s.size(2))
            grads_s = grads_s.permute(0, 3, 1, 2)  # [batch_size, H, W, C] -> [batch, C, H, W]

        grads_detach_s = grads_s.detach()  # [B, C, H, W]
        weights_s = torch.mean(grads_detach_s, [2, 3], keepdim=True)  # [B, C, 1, 1]
        cam_s = torch.sum(weights_s * stage4_feature_s, dim=1)  # [B, H, W]
        cam_s = torch.relu(cam_s)
        # 应用最小最大归一化
        cam_s = cam_s - cam_s.reshape(batch_size, -1).min(dim=1, keepdim=True)[0].reshape(batch_size, 1, 1)
        cam_s = cam_s / (cam_s.reshape(batch_size, -1).max(dim=1, keepdim=True)[0].reshape(batch_size, 1, 1) + 1e-7)
        # 计算激活图对齐损失
        loss_act_align = F.smooth_l1_loss(cam_s, cam_t.detach())

        # 将模型的BN层设置为train()状态，与正常训练保持一致
        if isinstance(model_s, (ResNet, ResNet1)):
            model_s = train_bn(model_s)
        elif isinstance(model_s, (ConvNeXt, SwinTransformer)):
            model_s = activate_drop_path(model_s)

        # 求交叉熵损失函数
        outputs = model_s(images)  # batch_size, num_class
        loss_ce = criterion_ce(outputs, labels)

        # 求蒸馏的kl损失
        # 计算kl蒸馏损失
        loss_kl = criterion_kl(outputs, logits_t.detach())
        loss = args.am_weight * loss_act_align + args.kl_weight * loss_kl + args.ce_weight * loss_ce

        # 根据cosine scheduler 修改optimizer的learning rate 和 weight decay
        global_step = ith_epoch * num_ite_per_epoch + j
        for _, param_group in enumerate(optimizer.param_groups):
            if lr_schedule_values is not None:
                param_group["lr"] = lr_schedule_values[global_step]
            if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_schedule_values[global_step]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        align_losses.update(loss_act_align.item() * args.am_weight, batch_size)
        ce_losses.update(loss_ce.item() * args.ce_weight, batch_size)
        kl_losses.update(loss_kl.item() * args.kl_weight, batch_size)
        losses.update(loss.item(), batch_size)
        top1.update(metrics.accuracy(output=outputs, target=labels, topk=(1,))[0].item(), batch_size)
        batch_time.update(time.time() - tm)
        tm = time.time()
        # 记录结束时间
        end_time = time.time()

        # 计算运行时间
        elapsed_time = end_time - start_time

        print(f"代码运行时间: {elapsed_time:.4f} 秒")
        if j % 60 == 0:
            print(
                f'Train Epoch: {ith_epoch + 1} [{(j + 1) * batch_size}/{len(train_loader.dataset)} ({100. * ((j + 1) * batch_size) / len(train_loader.dataset) :.2f}%)]\
                             Align_Loss: {align_losses.val:.4f} (Avg:{align_losses.avg:.4f})\
                             Ce_Loss: {ce_losses.val:.4f} (Avg:{ce_losses.avg:.4f})\
                             KL_Loss: {kl_losses.val:.4f} (Avg:{kl_losses.avg:.4f})\
                             Loss: {losses.val:.4f} (Avg:{losses.avg:.4f})\
                             AccTop1: {top1.val:.3f} (Avg:{top1.avg:.3f})\
                             Time: {batch_time.val:.3f} (Avg:{batch_time.avg:.3f})')

    print(f'the {ith_epoch + 1} epoch training time {batch_time.sum:.3f}\n')
    return losses.avg


def train_distill_label_for_eigen_cam(train_loader, module_list, criterion_list, optimizer, args,
                                      num_ite_per_epoch, lr_schedule_values,
                                      wd_schedule_values, ith_epoch):
    """
    这个函数使用对蒸馏训练的一个epoch进行
    One epoch distillation
    """
    # 设置学生模型为训练模式
    module_list[0].train()
    # 设置教师模型为eval()模式
    module_list[1].eval()

    model_s = module_list[0]
    model_t = module_list[1]

    criterion_ce = criterion_list[0]
    criterion_kl = criterion_list[1]

    batch_time = DataUpdater()
    losses = DataUpdater()
    ce_losses = DataUpdater()
    kl_losses = DataUpdater()
    align_losses = DataUpdater()
    top1 = DataUpdater()
    metrics = Matrics()
    tm = time.time()
    device = next(model_s.parameters()).device

    for j, data in enumerate(train_loader):
        images, labels, _, _, image_paths = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.shape[0]
        image_height, image_width = images.size(2), images.size(3)
        # 首先利用反向传播求出教师模型的Activation map
        stage4_feature_t = model_t.forward_features(images)  # [B, C_t, H, W]
        logits_t = model_t.forward_head(stage4_feature_t)  # [B, num_class]
        # 这个地方要做判断是否为swin-transformer，从而对梯度和特征图进行reshape操作
        if isinstance(model_t, SwinTransformer):
            h, w = feature_size(image_height), feature_size(image_width)
            # 将特征图进行reshape操作
            stage4_feature_t = stage4_feature_t.reshape(stage4_feature_t.size(0), h, w, stage4_feature_t.size(2))
            stage4_feature_t = stage4_feature_t.permute(0, 3, 1, 2)  # [batch_size, H, W, C] -> [batch, C, H, W]
        cam_t = get_2d_projection(stage4_feature_t.detach())
        # 用0将负值截断，其实就是relu
        # cam_t = torch.maximum(cam_t, 0)
        cam_t = torch.clamp(cam_t, min=0)
        # 应用最小最大归一化
        cam_t = cam_t - cam_t.reshape(batch_size, -1).min(dim=1, keepdim=True)[0].reshape(batch_size, 1, 1)
        cam_t = cam_t / (cam_t.reshape(batch_size, -1).max(dim=1, keepdim=True)[0].reshape(batch_size, 1, 1) + 1e-7)

        # ================下面求学生模型的梯度======================

        if isinstance(model_s, (ResNet, ResNet1)):
            # 将模型的Batch Normalization设置为eval()，保证与推理过程中一致
            model_s = eval_bn(model_s)
        elif isinstance(model_s, (ConvNeXt, SwinTransformer)):
            model_s = deactivate_drop_path(model_s)

        stage4_feature_s = model_s.forward_features(images)  # [B, C_s, H, W]
        # 这个地方要做判断是否为swin-transformer，从而对梯度和特征图进行reshape操作
        if isinstance(model_s, SwinTransformer):
            h, w = feature_size(image_height), feature_size(image_width)
            # 将特征图进行reshape操作
            stage4_feature_s = stage4_feature_s.reshape(stage4_feature_s.size(0), h, w, stage4_feature_s.size(2))
            stage4_feature_s = stage4_feature_s.permute(0, 3, 1, 2)  # [batch_size, H, W, C] -> [batch, C, H, W]

        cam_s = get_2d_projection(stage4_feature_s)
        # 用0将负值截断，其实就是relu
        ## cam_s = torch.maximum(cam_s, 0)
        cam_s = torch.clamp(cam_s, min=0)
        # 应用最小最大归一化
        cam_s = cam_s - cam_s.reshape(batch_size, -1).min(dim=1, keepdim=True)[0].reshape(batch_size, 1, 1)
        cam_s = cam_s / (cam_s.reshape(batch_size, -1).max(dim=1, keepdim=True)[0].reshape(batch_size, 1, 1) + 1e-7)
        # 计算激活图对齐损失
        loss_act_align = F.smooth_l1_loss(cam_s, cam_t.detach())
        # 将模型的BN层设置为train()状态，与正常训练保持一致
        if isinstance(model_s, (ResNet, ResNet1)):
            model_s = train_bn(model_s)
        elif isinstance(model_s, (ConvNeXt, SwinTransformer)):
            model_s = activate_drop_path(model_s)

        # 求交叉熵损失函数
        outputs = model_s(images)  # batch_size, num_class
        loss_ce = criterion_ce(outputs, labels)

        # 求蒸馏的kl损失
        # 计算kl蒸馏损失
        loss_kl = criterion_kl(outputs, logits_t.detach())
        loss = args.am_weight * loss_act_align + args.kl_weight * loss_kl + args.ce_weight * loss_ce

        # 根据cosine scheduler 修改optimizer的learning rate 和 weight decay
        global_step = ith_epoch * num_ite_per_epoch + j
        for _, param_group in enumerate(optimizer.param_groups):
            if lr_schedule_values is not None:
                param_group["lr"] = lr_schedule_values[global_step]
            if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_schedule_values[global_step]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        align_losses.update(loss_act_align.item() * args.am_weight, batch_size)
        ce_losses.update(loss_ce.item() * args.ce_weight, batch_size)
        kl_losses.update(loss_kl.item() * args.kl_weight, batch_size)
        losses.update(loss.item(), batch_size)
        top1.update(metrics.accuracy(output=outputs, target=labels, topk=(1,))[0].item(), batch_size)
        batch_time.update(time.time() - tm)
        tm = time.time()
        if j % 60 == 0:
            print(
                f'Train Epoch: {ith_epoch + 1} [{(j + 1) * batch_size}/{len(train_loader.dataset)} ({100. * ((j + 1) * batch_size) / len(train_loader.dataset) :.2f}%)]\
                             Align_Loss: {align_losses.val:.4f} (Avg:{align_losses.avg:.4f})\
                             Ce_Loss: {ce_losses.val:.4f} (Avg:{ce_losses.avg:.4f})\
                             KL_Loss: {kl_losses.val:.4f} (Avg:{kl_losses.avg:.4f})\
                             Loss: {losses.val:.4f} (Avg:{losses.avg:.4f})\
                             AccTop1: {top1.val:.3f} (Avg:{top1.avg:.3f})\
                             Time: {batch_time.val:.3f} (Avg:{batch_time.avg:.3f})')

    print(f'the {ith_epoch + 1} epoch training time {batch_time.sum:.3f}\n')
    return losses.avg

# def train_distill_label(train_loader, module_list, criterion_list, optimizer, args,
#                         num_ite_per_epoch, lr_schedule_values,
#                         wd_schedule_values, ith_epoch):
#     """
#     这个函数使用对蒸馏训练的一个epoch进行
#     One epoch distillation
#     """
#     # 设置学生模型为训练模式
#     module_list[0].train()
#     # 设置教师模型为eval()模式
#     module_list[1].eval()
#
#     model_s = module_list[0]
#     model_t = module_list[1]
#
#     criterion_ce = criterion_list[0]
#     criterion_kl = criterion_list[1]
#
#     batch_time = DataUpdater()
#     losses = DataUpdater()
#     ce_losses = DataUpdater()
#     kl_losses = DataUpdater()
#     align_losses = DataUpdater()
#     top1 = DataUpdater()
#     metrics = Matrics()
#     tm = time.time()
#     device = next(model_s.parameters()).device
#
#     for j, data in enumerate(train_loader):
#         # 这个地方的activation_maps便是我们自定义的激活图
#         images, labels, activation_maps, mask_weights, image_paths = data
#         batch_size = len(labels)
#         images = images.to(device, non_blocking=True)
#         # 将图片输入到教师模型中，得到输出的logits, 用于蒸馏的kl散度的计算
#         logits_t = model_t(images)  # batch_size, num_class
#
#         if isinstance(model_s, (ResNet, ResNet1)):
#             # 将模型的Batch Normalization设置为eval()，保证与推理过程中一致
#             model_s = eval_bn(model_s)
#         elif isinstance(model_s, ConvNeXt):
#             model_s = deactivate_drop_path(model_s)
#
#         stage4_feature_s = model_s.forward_features(images)  # [B, C_s, H, W]
#         logits_s = model_s.forward_head(stage4_feature_s)  # [B, num_class]
#         # 接下来根据对应类别的logit反向传播求stage4_feature_s的梯度
#         # 计算每个样本的目标类别对应的梯度
#         optimizer.zero_grad()
#         target = torch.zeros_like(logits_s)
#         for i in range(batch_size):
#             target[i, int(labels[i])] = 1
#         logits_s.backward(gradient=target, retain_graph=True)
#
#         grads_detach = grads.detach()  # [B, C, H, W]
#         weights = torch.mean(grads_detach, [2, 3], keepdim=True)  # [B, C, 1, 1]
#         cam = torch.sum(weights * stage4_feature_s, dim=1)  # [B, H, W]
#         cam = torch.relu(cam)
#         # 应用最小最大归一化
#         cam = cam - cam.reshape(batch_size, -1).min(dim=1, keepdim=True)[0].reshape(batch_size, 1, 1)
#         cam = cam / (cam.reshape(batch_size, -1).max(dim=1, keepdim=True)[0].reshape(batch_size, 1, 1) + 1e-7)
#         # 将模型生成的cam激活图通过双线性插值的方法进行32倍的上采样
#         cam = upsample(cam, activation_maps.shape[1], activation_maps.shape[2])
#
#         # 计算生成的激活图的自定义的激活图的对齐损失
#         activation_maps = activation_maps.to(device, non_blocking=True)
#         loss_act_align = F.smooth_l1_loss(cam, activation_maps, reduction='none')  # [B, H, W]
#         # 这个地方应该根据分割掩膜的类别数量求得每一个位置的权重,对每一个位置的损失加权之后再取平均
#         mask_weights = mask_weights.to(device, non_blocking=True)  # [B, H, W]
#         loss_act_align = (loss_act_align * mask_weights).mean()
#
#         # 将模型的BN层设置为train()状态，与正常训练保持一致
#         if isinstance(model_s, (ResNet, ResNet1)):
#             model_s = train_bn(model_s)
#         elif isinstance(model_s, ConvNeXt):
#             model_s = activate_drop_path(model_s)
#
#         outputs = model_s(images)  # batch_size, num_class
#         labels = labels.to(device, non_blocking=True)
#         loss_ce = criterion_ce(outputs, labels)
#
#         # 求蒸馏的kl损失
#         # 计算kl蒸馏损失
#         loss_kl = criterion_kl(outputs, logits_t.detach())
#         loss = args.am_weight * loss_act_align + args.kl_weight * loss_kl + args.ce_weight * loss_ce
#
#         # 根据cosine scheduler 修改optimizer的learning rate 和 weight decay
#         global_step = ith_epoch * num_ite_per_epoch + j
#         for _, param_group in enumerate(optimizer.param_groups):
#             if lr_schedule_values is not None:
#                 param_group["lr"] = lr_schedule_values[global_step]
#             if wd_schedule_values is not None and param_group["weight_decay"] > 0:
#                 param_group["weight_decay"] = wd_schedule_values[global_step]
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         align_losses.update(loss_act_align.item() * args.am_weight, batch_size)
#         ce_losses.update(loss_ce.item() * args.ce_weight, batch_size)
#         kl_losses.update(loss_kl.item() * args.kl_weight, batch_size)
#         losses.update(loss.item(), batch_size)
#         top1.update(metrics.accuracy(output=outputs, target=labels, topk=(1,))[0].item(), batch_size)
#         batch_time.update(time.time() - tm)
#         tm = time.time()
#         if j % 20 == 0:
#             print(
#                 f'Train Epoch: {ith_epoch + 1} [{(j + 1) * batch_size}/{len(train_loader.dataset)} ({100. * ((j + 1) * batch_size) / len(train_loader.dataset) :.2f}%)]\
#                              Align_Loss: {align_losses.val:.4f} (Avg:{align_losses.avg:.4f})\
#                              Ce_Loss: {ce_losses.val:.4f} (Avg:{ce_losses.avg:.4f})\
#                              KL_Loss: {kl_losses.val:.4f} (Avg:{kl_losses.avg:.4f})\
#                              Loss: {losses.val:.4f} (Avg:{losses.avg:.4f})\
#                              AccTop1: {top1.val:.3f} (Avg:{top1.avg:.3f})\
#                              Time: {batch_time.val:.3f} (Avg:{batch_time.avg:.3f})')
#
#     print(f'the {i + 1} epoch training time {batch_time.sum:.3f}\n')
#     return losses.avg
