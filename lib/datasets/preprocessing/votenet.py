import glob
import os

import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm

ROOT_GLOB = '/home/jgwak/SourceCodes/votenet/scannet/scannet_train_detection_data2/*_vert.npy'
OUT_DIR = 'votenet_scannet_rgb'
INST_EXT = '_ins_label.npy'
SEM_EXT = '_sem_label.npy'

os.mkdir(OUT_DIR)
for vert_f in tqdm(glob.glob(ROOT_GLOB)):
  scan_id = vert_f[:-9]
  inst_f = scan_id + INST_EXT
  sem_f = scan_id + SEM_EXT
  assert os.path.isfile(inst_f) and os.path.isfile(sem_f)
  vert = np.load(vert_f)
  inst = np.load(inst_f)
  sem = np.load(sem_f)
  python_types = (float, float, float, int, int, int, int, int)
  npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
               ('label_class', 'u1'), ('label_instance', 'u2')]
  vertices = []
  for row_idx in range(vert.shape[0]):
    cur_point = np.concatenate((vert[row_idx], np.array((sem[row_idx], inst[row_idx]))))
    vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
  vertices_array = np.array(vertices, dtype=npy_types)
  el = PlyElement.describe(vertices_array, 'vertex')
  PlyData([el]).write(f'{OUT_DIR}/{scan_id.split(os.sep)[-1]}.ply')
