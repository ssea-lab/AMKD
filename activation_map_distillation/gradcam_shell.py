import os

test_command = 'python gradcam.py --model_t={} --model_s={} --gpu=1 --batch_size=1 --use_label_smoothing --use_class_weight --pre_train --test_file={} --use_am --use_kl --use_ce --cam_type={}'
t_s_dict = {
    'resnet18': ['resnet_little'],
    'convnext_pico': ['convnext_little'],
    'swin_tiny': ['swin_little']
}
model_list = ['resnet18', 'convnext_pico', 'swin_tiny']
label_list = [' --use_unlabel_data --use_pselabel --use_label_data', ' --use_label_data']
for model_t in model_list:
    model_s = t_s_dict[model_t][0]
    for test_file in ['internal_test', 'tiff_huaxi', 'tiff_xiangya']:
        for label in label_list:
            cam_type_list = ['gradcam', 'eigen_cam'] if label == ' --use_label_data' else ['gradcam']
            for cam_type in cam_type_list:
                test_command1 = test_command.format(model_t, model_s, test_file, cam_type)
                test_command2 = test_command1 + label
                print(test_command2)
                os.system(test_command2)
