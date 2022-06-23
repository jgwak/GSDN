import os
import os.path as osp
import sys
import numpy as np
import MinkowskiEngine as ME
import multiprocessing as mp
from functools import partial

from lib.pc_utils import Camera, save_point_cloud
from lib.datasets.synthia import Synthia2dDataset

SYNTHIA_PATH = '/cvgl/group/Synthia'
SYNTHIA_OUT_PATH = '/cvgl2/u/cornman/Synthia/synthia-processed/'

SYNTHIA_INTRINSICS_F = 'CameraParams/intrinsics.txt'
SYNTHIA_LABEL_DIR = 'GT/LABELS'
SYNTHIA_DEPTH_DIR = 'Depth'
SYNTHIA_RGB_DIR = 'RGB'
SYNTHIA_CAMERA_PARAMS_DIR = 'CameraParams'
SYNTHIA_EXTRINSICS_FRONT = 'Omni_F'

STEREO_LIST = ['Stereo_Left', 'Stereo_Right']
OMNI_LIST = ['Omni_B', 'Omni_F', 'Omni_L', 'Omni_R']
NUM_BATCH_CHUNKS = 8

SYNTHIA_TRAINSET = (1, 2, 3, 4)
SYNTHIA_VALSET = (5,)
SYNTHIA_TESTSET = (6,)

SUBSAMPLE_VOXEL_SIZE = 5  # 5cm
BACKPROJ_MAX_DEPTH = 5000  # 50m

SYNTHIA_DIRS_SHUFFLE = [
    'SYNTHIA-SEQS-06-WINTER', 'SYNTHIA-SEQS-01-FALL', 'SYNTHIA-SEQS-05-SPRING',
    'SYNTHIA-SEQS-05-RAIN', 'SYNTHIA-SEQS-06-SUNSET', 'SYNTHIA-SEQS-06-SPRING',
    'SYNTHIA-SEQS-01-SPRING', 'SYNTHIA-SEQS-02-RAINNIGHT', 'SYNTHIA-SEQS-06-DAWN',
    'SYNTHIA-SEQS-04-SUMMER', 'SYNTHIA-SEQS-04-SUNSET', 'SYNTHIA-SEQS-02-SPRING',
    'SYNTHIA-SEQS-06-FOG', 'SYNTHIA-SEQS-02-FALL', 'SYNTHIA-SEQS-05-WINTER',
    'SYNTHIA-SEQS-04-RAINNIGHT', 'SYNTHIA-SEQS-05-FALL', 'SYNTHIA-SEQS-05-NIGHT',
    'SYNTHIA-SEQS-04-SPRING', 'SYNTHIA-SEQS-01-FOG', 'SYNTHIA-SEQS-05-FOG', 'SYNTHIA-SEQS-01-DAWN',
    'SYNTHIA-SEQS-04-FOG', 'SYNTHIA-SEQS-05-RAINNIGHT', 'SYNTHIA-SEQS-02-WINTER',
    'SYNTHIA-SEQS-05-SUMMER', 'SYNTHIA-SEQS-02-SUMMER', 'SYNTHIA-SEQS-02-NIGHT',
    'SYNTHIA-SEQS-02-SUNSET', 'SYNTHIA-SEQS-04-WINTER', 'SYNTHIA-SEQS-02-FOG',
    'SYNTHIA-SEQS-01-WINTERNIGHT', 'SYNTHIA-SEQS-06-SUMMER', 'SYNTHIA-SEQS-04-SOFTRAIN',
    'SYNTHIA-SEQS-01-SUMMER', 'SYNTHIA-SEQS-05-SUNSET', 'SYNTHIA-SEQS-02-DAWN',
    'SYNTHIA-SEQS-04-DAWN', 'SYNTHIA-SEQS-02-SOFTRAIN', 'SYNTHIA-SEQS-01-WINTER',
    'SYNTHIA-SEQS-04-FALL', 'SYNTHIA-SEQS-06-WINTERNIGHT', 'SYNTHIA-SEQS-01-NIGHT',
    'SYNTHIA-SEQS-05-WINTERNIGHT', 'SYNTHIA-SEQS-06-NIGHT', 'SYNTHIA-SEQS-05-SOFTRAIN',
    'SYNTHIA-SEQS-04-NIGHT', 'SYNTHIA-SEQS-01-SUNSET', 'SYNTHIA-SEQS-04-WINTERNIGHT',
    'SYNTHIA-SEQS-05-DAWN'
]


def force_print(msg):
  """Dirty workaround for slurm jobs not flushing buffers frequently."""
  print(msg)
  sys.stdout.flush()


def generate_and_save_ply(out_path, data_file_lists, num_cameras, intrinsics, i):
  force_print('Processing frame: %04d' % i)
  camera = Camera(intrinsics)
  all_points = []
  extrinsics_front = []
  for cam in range(num_cameras):
    data_tuple = data_file_lists[cam]
    rgb_file, depth_file, label_file, extrinsics_file = [data[i] for data in data_tuple]
    basename = osp.basename(rgb_file)
    basename = osp.splitext(basename)[0]  # remove extension
    extrinsics = Synthia2dDataset.load_extrinsics(extrinsics_file)
    if SYNTHIA_EXTRINSICS_FRONT in extrinsics_file:
      extrinsics_front.append(extrinsics)
    rgb = Synthia2dDataset.load_rgb(rgb_file)
    # NOTE: There exists corrupt depth files from upstream. We will ignore them.
    try:
      depth = Synthia2dDataset.load_depth(depth_file)
    except ValueError:
      continue
    label = Synthia2dDataset.load_label(label_file)
    points3d, labels3d = camera.backproject(
        depth, labels=label, rgb_img=rgb, extrinsics=extrinsics, max_depth=BACKPROJ_MAX_DEPTH)
    labels = labels3d[:, 3:-1]  # remove xyz and last channel
    points3d = np.hstack((points3d, labels))
    all_points.append(points3d)
  all_points = np.vstack(all_points)
  xyz = all_points[:, :3]
  unique_indices = ME.SparseVoxelize(xyz / SUBSAMPLE_VOXEL_SIZE, return_index=True)
  all_points = all_points[unique_indices]
  assert len(extrinsics_front) == 2
  assert np.allclose(extrinsics_front[0][:3, :3], extrinsics_front[1][:3, :3])
  extrinsics_front = np.mean(extrinsics_front, 0)
  all_points[:, :3] = camera.world2camera(extrinsics_front, all_points[:, :3])
  all_points[:, 1:3] *= -1  # invert y and z coordinate
  out_file = osp.join(out_path, basename + '.ply')
  save_point_cloud(all_points, out_file, with_label=True)


def write_data_ply(synthia_root_dir, synthia_out_path, batch_index, num_chunks):
  """Write ply files with xyz, rgb, semantic label, instance label

  Assumptions:
  - All directories have the same format and structure as the SYNTHIA-SEQS-01-DAWN seq.
  - The number of frames for all data (depth/rgb/labels + stereo_right/stereo_left + omni F/B/L/R)
    is the same among all combinations of the three.

  Example root dir: /cvgl/group/Synthia
  """
  chunk = np.array_split(SYNTHIA_DIRS_SHUFFLE, num_chunks)[batch_index].tolist()
  force_print('Batch Job: ' + ', '.join(chunk))
  for seq_id in chunk:
    force_print('Processing: ' + seq_id)
    seq_dir = osp.join(synthia_root_dir, seq_id)
    intrinsics_f = osp.join(seq_dir, SYNTHIA_INTRINSICS_F)
    intrinsics = Synthia2dDataset.load_intrinsics(intrinsics_f)

    label_dir = osp.join(seq_dir, SYNTHIA_LABEL_DIR)
    depth_dir = osp.join(seq_dir, SYNTHIA_DEPTH_DIR)
    rgb_dir = osp.join(seq_dir, SYNTHIA_RGB_DIR)
    cam_params_dir = osp.join(seq_dir, SYNTHIA_CAMERA_PARAMS_DIR)

    num_cameras = len(STEREO_LIST) * len(OMNI_LIST)
    # data_file_lists is list of tuple of lists
    data_file_lists = []  # populate with one tuple per camera
    num_frames = 0

    for stereo in STEREO_LIST:
      for omni in OMNI_LIST:
        rgb_deq_dir = osp.join(rgb_dir, stereo, omni)
        depth_seq_dir = osp.join(depth_dir, stereo, omni)
        label_seq_dir = osp.join(label_dir, stereo, omni)
        extrinsics_seq_dir = osp.join(cam_params_dir, stereo, omni)

        rgb_files = sorted(
            [osp.join(rgb_deq_dir, rgb) for rgb in os.listdir(rgb_deq_dir) if rgb.endswith('.png')])
        depth_files = sorted([
            osp.join(depth_seq_dir, depth)
            for depth in os.listdir(depth_seq_dir)
            if depth.endswith('.png')
        ])
        label_files = sorted([
            osp.join(label_seq_dir, label)
            for label in os.listdir(label_seq_dir)
            if label.endswith('.png')
        ])
        extrinsincs_files = sorted([
            osp.join(extrinsics_seq_dir, extr)
            for extr in os.listdir(extrinsics_seq_dir)
            if extr.endswith('.txt')
        ])

        assert (len(rgb_files) == len(depth_files) == len(label_files) == len(extrinsincs_files))
        files_tuple = (rgb_files, depth_files, label_files, extrinsincs_files)
        data_file_lists.append(files_tuple)
        num_frames = len(rgb_files)

    out_path = os.path.join(synthia_out_path, seq_id)
    if not osp.exists(out_path):
      os.makedirs(out_path)
    frames = list(range(num_frames))
    p = mp.Pool(processes=mp.cpu_count())
    func = partial(generate_and_save_ply, out_path, data_file_lists, num_cameras, intrinsics)
    p.map(func, frames)
    p.close()
    p.join()


if __name__ == '__main__':
  batch_index = int(sys.argv[1])
  write_data_ply(SYNTHIA_PATH, SYNTHIA_OUT_PATH, batch_index, NUM_BATCH_CHUNKS)
