import time

import numpy as np
import open3d as o3d
import torch
import MinkowskiEngine as ME

import lib.pc_utils as pc_utils
from config import get_config
from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset
from lib.pipelines import load_pipeline
from lib.voxelizer import Voxelizer

INPUT_PCD = 'outputs/stanford_building5.ply'
LOCFEAT_IDX = 2
MIN_CONF = 0.9

from pytorch_memlab import profile

@profile
def main():
  pcd = o3d.io.read_point_cloud(INPUT_PCD)
  pcd_xyz, pcd_feats = np.asarray(pcd.points), np.asarray(pcd.colors)
  print(f'Finished reading {INPUT_PCD}:')
  print(f'# points: {pcd_xyz.shape[0]} points')
  print(f'volume: {np.prod(pcd_xyz.max(0) - pcd_xyz.min(0))} m^3')

  sparse_voxelizer = Voxelizer(voxel_size=0.05)

  height = pcd_xyz[:, LOCFEAT_IDX].copy()
  height -= np.percentile(height, 0.99)
  pcd_feats = np.hstack((pcd_feats, height[:, None]))

  preprocess = []
  for i in range(7):
    start = time.time()
    coords, feats, labels, transformation = sparse_voxelizer.voxelize(pcd_xyz, pcd_feats, None)
    preprocess.append(time.time() - start)
  print('Voxelization time average: ', np.mean(preprocess[2:]))

  coords = ME.utils.batched_coordinates([torch.from_numpy(coords).int()])
  feats = torch.from_numpy(feats.astype(np.float32)).to('cuda')

  config = get_config()
  DatasetClass = load_dataset(config.dataset)
  dataloader = initialize_data_loader(
      DatasetClass,
      config,
      threads=config.threads,
      phase=config.test_phase,
      augment_data=False,
      shuffle=False,
      repeat=False,
      batch_size=config.test_batch_size,
      limit_numpoints=False)
  pipeline_model = load_pipeline(config, dataloader.dataset)
  if config.weights.lower() != 'none':
    state = torch.load(config.weights)
    pipeline_model.load_state_dict(state['state_dict'], strict=(not config.lenient_weight_loading))

  pipeline_model.eval()

  sinput = ME.SparseTensor(feats, coords).to('cuda')
  datum = {'sinput': sinput, 'anchor_match_coords': None}
  evaltime = []
  for i in range(7):
    start = time.time()
    sinput = ME.SparseTensor(feats, coords).to('cuda')
    datum = {'sinput': sinput, 'anchor_match_coords': None}
    outputs = pipeline_model(datum, False)
    evaltime.append(time.time() - start)
  print('Network runtime average: ', np.mean(evaltime[2:]))

  pred = outputs['detection'][0]
  pred_mask = pred[:, -1] > MIN_CONF
  pred = pred[pred_mask]
  print(f'Detected {pred.shape[0]} instances')

  bbox_xyz = pred[:, :6]
  bbox_xyz += 0.5
  bbox_xyz[:, :3] += 0.5
  bbox_xyz[:, 3:] -= 0.5
  bbox_xyz[:, 3:] = np.maximum(bbox_xyz[:, 3:], bbox_xyz[:, :3] + 0.1)
  bbox_xyz = bbox_xyz.reshape(-1, 3)
  bbox_xyz1 = np.hstack((bbox_xyz, np.ones((bbox_xyz.shape[0], 1))))
  bbox_xyz = np.linalg.solve(transformation.reshape(4, 4), bbox_xyz1.T).T[:, :3].reshape(-1, 6)
  pred = np.hstack((bbox_xyz, pred[:, 6:]))
  pred_pcd = pc_utils.visualize_bboxes(pred[:, :6], pred[:, 6], num_points=100)

  mask = pcd_xyz[:, 2] < 2.3
  pc_utils.visualize_pcd(np.hstack((pcd_xyz[mask], pcd_feats[mask, :3])), pred_pcd)


if __name__ == '__main__':
  main()

# python scripts/stanford_full.py --scannet_votenetrgb_path /scr/jgwak/Datasets/scannet_votenet_rgb --scheduler ExpLR --exp_gamma 0.95 --max_iter 120000 --threads 6 --batch_size 16 --train_phase trainval --val_phase test --test_phase test --pipeline SparseGenerativeOneShotDetector --dataset ScannetVoteNetRGBDataset --load_sparse_gt_data true --backbone_model ResNet34 --sfpn_classification_loss balanced --sfpn_min_confidence 0.3 --rpn_anchor_ratios 0.25,0.25,4.0,0.25,4.0,0.25,0.25,4.0,4.0,4.0,0.25,0.25,4.0,0.25,4.0,4.0,4.0,0.25,0.5,0.5,2.0,0.5,2.0,0.5,0.5,2.0,2.0,2.0,0.5,0.5,2.0,0.5,2.0,2.0,2.0,0.5,1.0,1.0,1.0 --weights /cvgl2/u/jgwak/SourceCodes/MinkowskiDetection.new/outputs/scannetrgb_round2_ar124/weights.pth --is_train false --test_original_pointcloud true --return_transformation true --detection_min_confidence 0.1 --detection_nms_threshold 0.1 --normalize_bbox false --detection_max_instance 2000 --rpn_pre_nms_limit 100000
