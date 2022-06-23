import re
import os

COMMANDS = {
    'best': 'python main.py --scannet_votenetrgb_path /scr/jgwak/Datasets/scannet_votenet_rgb/ --threads 8 --fpn_max_scale 64 --batch_size 4 --train_phase train --val_phase val --test_phase val --scheduler PolyLR --max_iter 180000 --pipeline MaskRCNN_PointNet --heldout_save_freq 20000 --dataset ScannetVoteNetRGBDataset --weights /cvgl2/u/jgwak/SourceCodes/MinkowskiDetection/outputs/round10_inst_scannetvotenetrgb_train/weights.pth --is_train false',
}

for exp, command in COMMANDS.items():
  inst_csv = ''
  for det_confidence in ['0.0', '0.1', '0.2', '0.3']:
    for det_nms in ['0.2', '0.3', '0.35']:
      for mask_confidence in ['0.5']:
        for mask_nms in ['0.7', '0.8', '0.9', '1.0']:
          result_f = f'{exp}_dconf{det_confidence}_dnms{det_nms}_mconf{mask_confidence}_mnms{mask_nms}.txt'
          ap_inst = '-----'
          if os.path.isfile(result_f):
            with open(result_f) as f:
              result = [l.rstrip() for l in f.readlines()]
            last_result = ' '.join(result[-3:])
            parse_result = re.search(r'ap_inst:\s([0-9\.]+).*Finished.*', last_result)
            if parse_result is not None:
              ap_inst = parse_result.group(1)
          inst_csv += ap_inst + ','
        inst_csv += '    '
      inst_csv += '\n'
    inst_csv += '\n'
  print(inst_csv)
