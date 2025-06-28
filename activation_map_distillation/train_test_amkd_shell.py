import os

"""
这个脚本用来记录amkd蒸馏训练中所需要运行的脚本
"""
# 目前的话我们所采取的策略就是既使用大量无标签数据，又使用少量有标签数据进行蒸馏，没有必要单独进行有标签的蒸馏
label_list = [' --use_unlabel_data --use_pselabel --use_label_data']
t_s_dict = {
    'resnet18': ['resnet_little', 'resnet_lite'],
    'convnext_pico': ['convnext_little', 'convnext_lite'],
    'swin_tiny': ['swin_little', 'swin_lite']}
loss_list = [' --use_am --use_kl --use_ce ']

# command = "python train_amkd.py --model_t={} --model_s={} --gpu=1 --use_label_smoothing --pre_train --use_class_weight --batch_size=16"
# for model_t in ['swin_tiny']:
# # for model_t in ['resnet18', 'convnext_pico', 'swin_tiny']:
#     for model_s in t_s_dict[model_t]:
#         command_copy = command.format(model_t, model_s)
#         for label in label_list:
#             command_copy_1 = command_copy + label
#             for loss in loss_list:
#                 command_copy_2 = command_copy_1 + loss
#                 print(command_copy_2)
#                 os.system(command_copy_2)


# 既使用有标签的数据也使用无标签的数据进行蒸馏的测试
test_command = "python test_amkd.py --model_t={} --model_s={} --test_file={} --gpu=0 --use_label_smoothing"
for model_t in ['resnet18', 'convnext_pico', 'swin_tiny']:
    for model_s in t_s_dict[model_t]:
        for test_file in ['internal_test', 'tiff_huaxi', 'tiff_xiangya']:
            command_copy = test_command.format(model_t, model_s, test_file)
            for label in label_list:
                command_copy_1 = command_copy + label
                for loss in loss_list:
                    command_copy_2 = command_copy_1 + loss
                    print(command_copy_2)
                    os.system(command_copy_2)