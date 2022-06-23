import random

import numpy as np


COMMANDS = {
    'best': 'python main.py --scannet_votenetrgb_path /scr/jgwak/Datasets/scannet_votenet_rgb/ --threads 8 --fpn_max_scale 64 --batch_size 4 --train_phase train --val_phase val --test_phase val --scheduler PolyLR --max_iter 180000 --pipeline MaskRCNN_PointNet --heldout_save_freq 20000 --dataset ScannetVoteNetRGBDataset --weights /cvgl2/u/jgwak/SourceCodes/MinkowskiDetection/outputs/round10_inst_scannetvotenetrgb_train/weights.pth --is_train false',
}

all_commands = []
for exp, command in COMMANDS.items():
  for det_confidence in ['0.0', '0.1', '0.2', '0.3']:
    for det_nms in ['0.2', '0.3', '0.35']:
      for mask_confidence in ['0.5']:
        for mask_nms in ['0.7', '0.8', '0.9', '1.0']:
          all_commands.append(command + f' --detection_min_confidence {det_confidence} --detection_nms_threshold {det_nms} --mask_min_confidence {mask_confidence} --mask_nms_threshold {mask_nms} > {exp}_dconf{det_confidence}_dnms{det_nms}_mconf{mask_confidence}_mnms{mask_nms}.txt\n')

random.shuffle(all_commands)
command_splits = np.array_split(all_commands, 2)

with open('all_exps1.txt', 'w') as f:
  f.writelines(command_splits[0])

with open('all_exps2.txt', 'w') as f:
  f.writelines(['CUDA_VISIBLE_DEVICES=1 ' + l for l in command_splits[1]])
