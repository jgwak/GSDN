import os
import glob
import sys

import tqdm
import numpy as np

CLASSES = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa',
    'bookcase', 'board', 'clutter'
]
STANFORD_3D_PATH = '/scr/jgwak/Datasets/Stanford3dDataset_v1.2/Area_%d/*'
OUT_DIR = 'stanford3d'
CROP_SIZE = 5
MIN_POINTS = 10000
PREVOXELIZE_SIZE = 0.02


def read_pointcloud(filename):
  with open(filename) as f:
    ptc = np.array([line.rstrip().split() for line in f.readlines()])
  return ptc.astype(np.float32)


def subsample(ptc):
  voxel_coords = np.floor(ptc[:, :3] / PREVOXELIZE_SIZE).astype(int)
  _, unique_idxs = np.unique(voxel_coords, axis=0, return_index=True)
  return ptc[unique_idxs]


i = int(sys.argv[-1])
print(f'Processing Area {i}')
for room_path in tqdm.tqdm(glob.glob(STANFORD_3D_PATH % i)):
  if room_path.endswith('.txt'):
    continue
  room_name = room_path.split(os.sep)[-1]
  num_ptc = 0
  sem_labels = []
  inst_labels = []
  xyzrgb = []
  for j, instance_path in enumerate(glob.glob(f'{room_path}/Annotations/*')):
    instance_ptc = read_pointcloud(instance_path)
    instance_name = os.path.splitext(instance_path.split(os.sep)[-1])[0]
    instance_class = '_'.join(instance_name.split('_')[:-1])
    instance_idx = j
    try:
      class_idx = CLASSES.index(instance_class)
    except ValueError:
      if instance_class != 'stairs':
        raise
      print(f'Marking unknown class {instance_class} as ignore label.')
      class_idx = -1
      instance_idx = -1
    sem_labels.append(np.ones((instance_ptc.shape[0]), dtype=int) * class_idx)
    inst_labels.append(np.ones((instance_ptc.shape[0]), dtype=int) * instance_idx)
    xyzrgb.append(instance_ptc)
    num_ptc += instance_ptc.shape[0]
  all_ptc = np.hstack((np.vstack(xyzrgb), np.concatenate(sem_labels)[:, None],
                       np.concatenate(inst_labels)[:, None]))
  all_xyz = all_ptc[:, :3]
  all_xyz_min = all_xyz.min(0)
  room_size = all_xyz.max(0) - all_xyz_min

  if i != 5 and np.any(room_size > CROP_SIZE):  # Save Area5 as-is.
    k = 0
    steps = (np.floor(room_size / CROP_SIZE) * 2).astype(int) + 1
    for dx in range(steps[0]):
      for dy in range(steps[1]):
        for dz in range(steps[2]):
          crop_idx = np.array([dx, dy, dz])
          crop_min = crop_idx * CROP_SIZE / 2 + all_xyz_min
          crop_max = crop_min + CROP_SIZE
          crop_mask = np.all(np.hstack((crop_min < all_xyz, all_xyz < crop_max)), 1)
          if np.sum(crop_mask) < MIN_POINTS:
            continue
          crop_xyz = all_xyz[crop_mask]
          size_full = (crop_xyz.max(0) - crop_xyz.min(0)) > CROP_SIZE / 2
          init_dim = np.array([dx, dy, dz]) == 0
          if not np.all(np.logical_or(size_full, init_dim)):
            continue
          np.savez_compressed(f'{OUT_DIR}/Area{i}_{room_name}_{k}', subsample(all_ptc[crop_mask]))
          k += 1
  else:
    np.savez_compressed(f'{OUT_DIR}/Area{i}_{room_name}_0', subsample(all_ptc))
