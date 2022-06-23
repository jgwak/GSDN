import os

import numpy as np

import lib.pc_utils as pc_utils
from lib.utils import read_txt

OURS_PRED_PATH = 'outputs/visualization/ours_stanford'
STANFORD_PATH = '/cvgl/group/Stanford3dDataset_v1.2/Area_5'
NUM_BBOX_POINTS = 1000

files = sorted(read_txt('/scr/jgwak/Datasets/stanford3d/test.txt'))
for i, fn in enumerate(files):
  area_name = os.path.splitext(fn.split(os.sep)[-1])[0]
  area_name = '_'.join(area_name.split('_')[1:])[:-2]
  # area_name = 'office_34'
  # i = [i for i, fn in enumerate(files) if area_name in fn][0]
  if area_name.startswith('WC_') or area_name.startswith('hallway_'):
    continue
  ptc_fn = os.path.join(STANFORD_PATH, area_name, f'{area_name}.txt')
  ptc = np.array([l.split() for l in read_txt(ptc_fn)]).astype(float)
  pred_ours = np.load(os.path.join(OURS_PRED_PATH, 'out_%03d.npy.npz' % i))
  gt = pc_utils.visualize_bboxes(pred_ours['gt'][:, :6], pred_ours['gt'][:, 6],
                                 num_points=NUM_BBOX_POINTS)
  pred_ours = pc_utils.visualize_bboxes(pred_ours['pred'][:, :6], pred_ours['pred'][:, 6],
                                        num_points=NUM_BBOX_POINTS)
  params = pc_utils.visualize_pcd(gt, ptc, save_image=f'viz_Area5_{area_name}_gt.png')
  pc_utils.visualize_pcd(ptc, camera_params=params, save_image=f'viz_Area5_{area_name}_input.png')
  pc_utils.visualize_pcd(pred_ours, ptc, camera_params=params,
                         save_image=f'viz_Area5_{area_name}_ours.png')
