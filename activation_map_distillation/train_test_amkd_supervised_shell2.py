import os

"""
这个脚本用来记录amkd蒸馏训练中所需要运行的脚本
"""
# 目前的话我们所采取的策略就是既使用大量无标签数据，又使用少量有标签数据进行蒸馏，没有必要单独进行有标签的蒸馏
# label_list = [' --use_unlabel_data --use_pselabel --use_label_data']
label_list = [' --use_label_data']
# t_s_dict = {
#     'resnet18': ['resnet_little', 'resnet_lite'],
#     'convnext_pico': ['convnext_little', 'convnext_lite'],
#     'swin_tiny': ['swin_little', 'swin_lite']}
t_s_dict = {
    'resnet18': ['resnet_little'],
    'convnext_pico': ['convnext_little'],
    'swin_tiny': ['swin_little']}

loss_list = [' --use_am --use_kl --use_ce ']
# loss_list = [' --use_am --use_ce ']

# command = "python train_amkd_supervised.py --model_t={} --model_s={} --gpu=1 --use_label_smoothing --pre_train --use_class_weight --batch_size=16 --cam_type=eigen_cam "
# for model_t in ['swin_tiny']:
#     for model_s in t_s_dict[model_t]:
#         command_copy = command.format(model_t, model_s)
#         command_copy_1 = command_copy
#         for loss in loss_list:
#             command_copy_2 = command_copy_1 + loss
#             print(command_copy_2)
#             os.system(command_copy_2)


test_command = "python test.py --model_t={} --model_s={} --test_file={} --gpu=1 --use_label_smoothing --cam_type=eigen_cam"
for model_t in ['resnet18']:
    for model_s in t_s_dict[model_t]:
        for test_file in ['internal_test', 'tiff_huaxi', 'tiff_xiangya']:
            command_copy = test_command.format(model_t, model_s, test_file)
            for loss in loss_list:
                command_copy_1 = command_copy + loss
                print(command_copy_1)
                os.system(command_copy_1)