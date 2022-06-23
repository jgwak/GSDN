import os

import numpy as np
import open3d as o3d

import lib.pc_utils as pc_utils
from config import get_config
from lib.utils import read_txt

SCANNET_RAW_PATH = '/cvgl2/u/jgwak/Datasets/scannet_raw'
SCANNET_ALIGNMENT_PATH = '/cvgl2/u/jgwak/Datasets/scannet_raw/scans/%s/%s.txt'
VOTENET_PRED_PATH = 'outputs/visualization/votenet_scannet'
OURS_PRED_PATH = 'outputs/visualization/ours_scannet'
SIS_PRED_PATH = 'outputs/visualization/3dsis_scannet'
NUM_BBOX_POINTS = 1000

config = get_config()
files = sorted(read_txt('/scr/jgwak/Datasets/scannet_votenet_rgb/scannet_votenet_test.txt'))
for i, fn in enumerate(files):
  filename = fn.split(os.sep)[-1][:-4]
  if not os.path.isfile(os.path.join(SIS_PRED_PATH, f'{filename}.npz')):
    continue
  file_path = os.path.join(SCANNET_RAW_PATH, 'scans', filename, f"{filename}_vh_clean.ply")
  assert os.path.isfile(file_path)
  mesh = o3d.io.read_triangle_mesh(file_path)
  mesh.compute_vertex_normals()
  scene_f = SCANNET_ALIGNMENT_PATH % (filename, filename)
  alignment_txt = [l for l in read_txt(scene_f) if l.startswith('axisAlignment = ')][0]
  rot = np.array([float(x) for x in alignment_txt[16:].split()]).reshape(4, 4)
  mesh.transform(rot)
  pred_ours = np.load(os.path.join(OURS_PRED_PATH, 'out_%03d.npy.npz' % i))
  gt = pc_utils.visualize_bboxes(pred_ours['gt'][:, :6], pred_ours['gt'][:, 6],
                                 num_points=NUM_BBOX_POINTS)
  pred_ours = pc_utils.visualize_bboxes(pred_ours['pred'][:, :6], pred_ours['pred'][:, 6],
                                        num_points=NUM_BBOX_POINTS)
  params = pc_utils.visualize_pcd(gt, mesh, save_image=f'viz_{filename}_gt.png')
  pc_utils.visualize_pcd(mesh, camera_params=params, save_image=f'viz_{filename}_input.png')
  pc_utils.visualize_pcd(pred_ours, mesh, camera_params=params,
                         save_image=f'viz_{filename}_ours.png')
  pred_votenet = np.load(os.path.join(VOTENET_PRED_PATH, f'{filename}.npy'), allow_pickle=True)[0]
  votenet_preds = []
  for pred_cls, pred_bbox, pred_score in pred_votenet:
    pred_bbox = pred_bbox[:, (0, 2, 1)]
    pred_bbox[:, -1] *= -1
    votenet_preds.append(pc_utils.visualize_bboxes(np.expand_dims(pred_bbox, 0),
                         np.ones(1) * pred_cls, bbox_param='corners', num_points=NUM_BBOX_POINTS))
  votenet_preds = np.vstack(votenet_preds)
  pc_utils.visualize_pcd(votenet_preds, mesh, camera_params=params,
                         save_image=f'viz_{filename}_votenet.png')
  pred_3dsis = np.load(os.path.join(SIS_PRED_PATH, f'{filename}.npz'))
  pred_3dsis_corners = pred_3dsis['bbox_pred'].reshape(-1, 3)
  pred_3dsis_corners1 = np.hstack((pred_3dsis_corners, np.ones((pred_3dsis_corners.shape[0], 1))))
  pred_3dsis_rotcorners = (rot @ pred_3dsis_corners1.T).T[:, :3].reshape(-1, 8, 3)
  pred_3dsis_cls = pred_3dsis['bbox_cls'] - 1
  pred_3dsis_mask = pred_3dsis_cls > 0
  pred_3dsis_aligned = pc_utils.visualize_bboxes(pred_3dsis_rotcorners[pred_3dsis_mask],
                                                 pred_3dsis_cls[pred_3dsis_mask],
                                                 bbox_param='corners', num_points=NUM_BBOX_POINTS)
  pc_utils.visualize_pcd(mesh, pred_3dsis_aligned, camera_params=params,
                         save_image=f'viz_{filename}_3dsis.png')
