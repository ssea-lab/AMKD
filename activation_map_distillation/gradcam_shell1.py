import os

test_command = 'python gradcam.py --model_name={} --gpu=0 --batch_size=1 --use_label_smoothing --use_class_weight --pre_train --test_file={}'
model_list = ['resnet18']
for model_name in model_list:
    for test_file in ['tiff_huaxi']:
        test_command1 = test_command.format(model_name, test_file)
        print(test_command1)
        os.system(test_command1)
