import numpy as np
import random


COMMANDS = {
    'best': 'python main.py --scannet_votenet_path /scr/jgwak/Datasets/scannet_votenet --scheduler ExpLR --threads 4 --batch_size 8 --train_phase trainval --val_phase test --test_phase test --pipeline SparseGenerativeOneShotDetector --dataset ScannetVoteNetDataset --load_sparse_gt_data true --backbone_model ResNet34 --fpn_max_scale 64 --sfpn_classification_loss balanced --sfpn_min_confidence 0.1 --max_iter 120000 --exp_step_size 2745 --weights outputs/param_search_weight.pth --is_train false',
}

all_commands = []
for exp, command in COMMANDS.items():
  for confidence in np.linspace(0, 0.8, 5):
    for nms_threshold in np.linspace(0, 0.8, 5):
      for detection_nms_score in ('obj', 'sem', 'objsem'):
        for detection_ap_score in ('obj', 'sem', 'objsem'):
          for detection_max_instances in ('50', '100', '200'):
            all_commands.append(command + f' --detection_min_confidence {confidence:.1f} --detection_nms_threshold {nms_threshold:.1f} --detection_nms_score {detection_nms_score} --detection_ap_score {detection_ap_score} --detection_max_instances {detection_max_instances} > {exp}_conf{confidence:.1f}_nms{nms_threshold:.1f}_nms{detection_nms_score}_ap{detection_ap_score}_maxinst{detection_max_instances}.txt\n')

random.shuffle(all_commands)
for i, command in enumerate(all_commands):
  with open(f'all_exps{i % 4}.txt', 'a') as f:
    f.write(f'CUDA_VISIBLE_DEVICES={i % 2} ')
    f.write(command)
