import os

t_s_dict = {
    'resnet18': ['resnet_little'],
    'convnext_pico': ['convnext_little'],
    'swin_tiny': ['swin_little']}
dataset_list = ['tiff_huaxi', 'tiff_xiangya']
strategy_list = ['vote', 'continue_frames']
model_list = ['resnet18', 'convnext_pico', 'swin_tiny']
fold_num_dict = {'resnet_little_tiff_huaxi': 0, 'resnet_little_tiff_xiangya': 0,
                 'convnext_little_tiff_huaxi': 0, 'convnext_little_tiff_xiangya': 3,
                 'swin_little_tiff_huaxi': 4, 'swin_little_tiff_xiangya': 1}

command = 'python test_volume_vote.py --model_t={} --model_s={} --pre_train --gpu=0 --test_file={} --fold_num={} --strategy={} --voting_number={} --continue_pos_frames={} --use_unlabel_data --use_pselabel --use_label_data  --use_am --use_kl --use_ce '
for model_t in model_list:
    for model_s in t_s_dict[model_t]:
        for test_file in dataset_list:
            fold_num = fold_num_dict[f'{model_s}_{test_file}']
            for strategy in strategy_list:
                if strategy == 'vote':
                    for voting_number in range(2, 5):
                        command1 = command.format(model_t, model_s, test_file, fold_num, strategy, str(voting_number),  '0')
                        print(command1)
                        os.system(command1)
                elif strategy == 'continue_frames':
                    for continue_pos_frames in [2, 3, 4]:
                        command2 = command.format(model_t, model_s, test_file, fold_num, strategy, '0', str(continue_pos_frames))
                        os.system(command2)



