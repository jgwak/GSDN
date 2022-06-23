import re
import os

import numpy as np

COMMANDS = {
    'best': 'python main.py --scannet_votenet_path /scr/jgwak/Datasets/scannet_votenet --scheduler ExpLR --threads 4 --batch_size 8 --train_phase trainval --val_phase test --test_phase test --pipeline SparseGenerativeOneShotDetector --dataset ScannetVoteNetDataset --load_sparse_gt_data true --backbone_model ResNet34 --fpn_max_scale 64 --sfpn_classification_loss balanced --sfpn_min_confidence 0.1 --max_iter 120000 --exp_step_size 2745 --weights outputs/param_search_weight.pth --is_train false',
}
AP25_THRESH = 0.56
AP50_THRESH = 0.31

log_25 = ''
log_50 = ''
best_params = []
for exp, command in COMMANDS.items():
  for confidence in np.linspace(0, 0.8, 5):
    for nms_threshold in np.linspace(0, 0.8, 5):
      for detection_nms_score in ('obj', 'sem', 'objsem'):
        for detection_ap_score in ('obj', 'sem', 'objsem'):
          for detection_max_instances in ('50', '100', '200'):
            result_f = f'{exp}_conf{confidence:.1f}_nms{nms_threshold:.1f}_nms{detection_nms_score}_ap{detection_ap_score}_maxinst{detection_max_instances}.txt'
            if os.path.isfile(result_f):
              with open(result_f) as f:
                result = [l.rstrip() for l in f.readlines()]
              ap50_parse_result = re.search(r'ap_50 mAP:\s([0-9\.]+)', ''.join(result[-5:-2]))
              ap25_parse_result = re.search(r'ap_25 mAP:\s([0-9\.]+)', ''.join(result[-8:-5]))
              if ap50_parse_result is not None and ap25_parse_result is not None:
                ap_50 = ap50_parse_result.group(1)
                ap_25 = ap25_parse_result.group(1)
                if float(ap_25) > AP25_THRESH and float(ap_50) > AP50_THRESH:
                  best_params.append((
                      (confidence, nms_threshold, detection_nms_score, detection_ap_score, detection_max_instances),
                      (ap_25, ap_50)))
              else:
                ap_50 = '??????'
                ap_25 = '??????'
            log_25 += f'{ap_25}, '
            log_50 += f'{ap_50}, '
          log_25 += ' '
          log_50 += ' '
        log_25 += '    '
        log_50 += '    '
      log_25 += '\n'
      log_50 += '\n'
    log_25 += '\n'
    log_50 += '\n'
  print('AP@25--------------')
  print(log_25)
  print('AP@50---------------')
  print(log_50)
  print('BEST----------------')
  for params, (ap_25, ap_50) in best_params:
    print(f"{','.join(str(i) for i in params)}: AP@0.25: {ap_25}, AP@0.50: {ap_50}")
