import json
import random
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from lib.pc_utils import read_plyfile

SCANNET_RAW_PATH = Path('/cvgl2/u/jgwak/Datasets/scannet_raw')
SCANNET_OUT_PATH = Path('/scr/jgwak/Datasets/scannet_inst')
TRAIN_DEST = 'train'
TEST_DEST = 'test'
SUBSETS = {TRAIN_DEST: 'scans', TEST_DEST: 'scans_test'}
POINTCLOUD_FILE = '_vh_clean_2.ply'
CROP_SIZE = 6.
TRAIN_SPLIT = 0.8
BUGS = {
    'train/scene0270_00.ply': 50,
    'train/scene0270_02.ply': 50,
    'train/scene0384_00.ply': 149,
}


# TODO: Modify lib.pc_utils.save_point_cloud to take npy_types as input.
def save_point_cloud(points_3d, filename):
  assert points_3d.ndim == 2
  assert points_3d.shape[1] == 8
  python_types = (float, float, float, int, int, int, int, int)
  npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
               ('blue', 'u1'), ('label_class', 'u1'), ('label_instance', 'u2')]
  # Format into NumPy structured array
  vertices = []
  for row_idx in range(points_3d.shape[0]):
    cur_point = points_3d[row_idx]
    vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
  vertices_array = np.array(vertices, dtype=npy_types)
  el = PlyElement.describe(vertices_array, 'vertex')

  # Write
  PlyData([el]).write(filename)


# Preprocess data.
for out_path, in_path in SUBSETS.items():
  phase_out_path = SCANNET_OUT_PATH / out_path
  phase_out_path.mkdir(parents=True, exist_ok=True)
  for f in (SCANNET_RAW_PATH / in_path).glob('*/*' + POINTCLOUD_FILE):
    # Load pointcloud file.
    out_f = phase_out_path / (f.name[:-len(POINTCLOUD_FILE)] + f.suffix)
    pointcloud = read_plyfile(f)
    num_points = pointcloud.shape[0]
    # Make sure alpha value is meaningless.
    assert np.unique(pointcloud[:, -1]).size == 1

    # Load label.
    segment_f = f.with_suffix('.0.010000.segs.json')
    segment_group_f = (f.parent / f.name[:-len(POINTCLOUD_FILE)]).with_suffix('.aggregation.json')
    semantic_f = f.parent / (f.stem + '.labels' + f.suffix)
    if semantic_f.is_file():
      # Load semantic label.
      semantic_label = read_plyfile(semantic_f)
      # Sanity check that the pointcloud and its label has same vertices.
      assert num_points == semantic_label.shape[0]
      assert np.allclose(pointcloud[:, :3], semantic_label[:, :3])
      semantic_label = semantic_label[:, -1]
      # Load instance label.
      with open(segment_f) as f:
        segment = np.array(json.load(f)['segIndices'])
      with open(segment_group_f) as f:
        segment_groups = json.load(f)['segGroups']
      assert segment.size == num_points
      inst_idx = np.zeros(num_points)
      for group_idx, segment_group in enumerate(segment_groups):
        for segment_idx in segment_group['segments']:
          inst_idx[segment == segment_idx] = group_idx + 1
    else:  # Label may not exist in test case.
      semantic_label = np.zeros(num_points)
      inst_idx = np.zeros(num_points)
    pointcloud_label = np.hstack((pointcloud[:, :6], semantic_label[:, None], inst_idx[:, None]))
    save_point_cloud(pointcloud_label, out_f)


# Split trainval data to train/val according to scene.
trainval_files = [f.name for f in (SCANNET_OUT_PATH / TRAIN_DEST).glob('*.ply')]
trainval_scenes = list(set(f.split('_')[0] for f in trainval_files))
random.shuffle(trainval_scenes)
num_train = int(len(trainval_scenes) * TRAIN_SPLIT)
train_scenes = trainval_scenes[:num_train]
val_scenes = trainval_scenes[num_train:]

# Collect file list for all phase.
train_files = [f'{TRAIN_DEST}/{f}' for f in trainval_files if any(s in f for s in train_scenes)]
val_files = [f'{TRAIN_DEST}/{f}' for f in trainval_files if any(s in f for s in val_scenes)]
test_files = [f'{TEST_DEST}/{f.name}' for f in (SCANNET_OUT_PATH / TEST_DEST).glob('*.ply')]

# Data sanity check.
assert not set(train_files).intersection(val_files)
assert all((SCANNET_OUT_PATH / f).is_file() for f in train_files)
assert all((SCANNET_OUT_PATH / f).is_file() for f in val_files)
assert all((SCANNET_OUT_PATH / f).is_file() for f in test_files)

# Write file list for all phase.
with open(SCANNET_OUT_PATH / 'train.txt', 'w') as f:
  f.writelines([f + '\n' for f in train_files])
with open(SCANNET_OUT_PATH / 'val.txt', 'w') as f:
  f.writelines([f + '\n' for f in val_files])
with open(SCANNET_OUT_PATH / 'test.txt', 'w') as f:
  f.writelines([f + '\n' for f in test_files])

# Fix bug in the data.
for ply_file, bug_index in BUGS.items():
  ply_path = SCANNET_OUT_PATH / ply_file
  pointcloud = read_plyfile(ply_path)
  bug_mask = pointcloud[:, -2] == bug_index
  print(f'Fixing {ply_file} bugged label {bug_index} x {bug_mask.sum()}')
  pointcloud[bug_mask, -2] = 0
  save_point_cloud(pointcloud, ply_path)
