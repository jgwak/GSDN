import collections
import os
import glob
import json
import yaml

import numpy as np
import open3d as o3d
import tqdm
from PIL import Image


DATASET_TRAIN_PATH = '/scr/jgwak/Datasets/jrdb/dataset'
DATASET_TEST_PATH = '/scr/jgwak/Datasets/jrdb/dataset'
IN_IMG_STITCHED_PATH = 'images/image_stitched/%s/%s.jpg'
IN_PTC_LOWER_PATH = 'pointclouds/lower_velodyne/%s/%s.pcd'
IN_PTC_UPPER_PATH = 'pointclouds/upper_velodyne/%s/%s.pcd'
IN_LABELS_3D = 'labels/labels_3d/*.json'
IN_CALIBRATION_F = 'calibration/defaults.yaml'

OUT_D = '/scr/jgwak/Datasets/jrdb_d15_n15'
DIST_T = 15
NUM_PTS_T = 15


def get_calibration(input_dir):
    with open(os.path.join(input_dir, IN_CALIBRATION_F)) as f:
        return yaml.safe_load(f)['calibrated']


def get_full_file_list(input_dir):
  def _filepath2filelist(path):
    return set(tuple(os.path.splitext(f)[0].split(os.sep)[-2:])
               for f in glob.glob(os.path.join(input_dir, path % ('*', '*'))))

  def _label2filelist(path, key='labels'):
    seq_dicts = []
    for json_f in glob.glob(os.path.join(input_dir, path)):
      with open(json_f) as f:
        labels = json.load(f)
      seq_name = os.path.basename(os.path.splitext(json_f)[0])
      seq_dicts.append({(seq_name, os.path.splitext(file_name)[0]): label
                        for file_name, label in labels[key].items()})
    return dict(collections.ChainMap(*seq_dicts))

  imgs = _filepath2filelist(IN_IMG_STITCHED_PATH)
  lower_ptcs = _filepath2filelist(IN_PTC_LOWER_PATH)
  upper_ptcs = _filepath2filelist(IN_PTC_UPPER_PATH)
  labels_3d = _label2filelist(IN_LABELS_3D)
  filelist = set.intersection(imgs, lower_ptcs, upper_ptcs, labels_3d.keys())

  return {f: labels_3d[f] for f in sorted(filelist)}


def _load_stitched_image(input_dir, seq_name, file_name):
  img_path = os.path.join(input_dir, IN_IMG_STITCHED_PATH % (seq_name, file_name))
  return Image.open(img_path)


def process_3d(input_dir, calib, seq_name, file_name, labels_3d, out_f):

  def _load_pointcloud(path, calib_key):
    ptc = np.asarray(
        o3d.io.read_point_cloud(os.path.join(input_dir, path % (seq_name, file_name))).points)
    ptc -= np.expand_dims(np.array(calib[calib_key]['translation']), 0)
    theta = -float(calib[calib_key]['rotation'][-1])
    rotation_matrix = np.array(
        ((np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta))))
    ptc[:, :2] = np.squeeze(
        np.matmul(rotation_matrix, np.expand_dims(ptc[:, :2], 2)))
    return ptc

  lower_ptc = _load_pointcloud(IN_PTC_LOWER_PATH, 'lidar_lower_to_rgb')
  upper_ptc = _load_pointcloud(IN_PTC_UPPER_PATH, 'lidar_upper_to_rgb')
  ptc = np.vstack((upper_ptc, lower_ptc))

  image = _load_stitched_image(input_dir, seq_name, file_name)
  ptc_rect = ptc[:, [1, 2, 0]]
  ptc_rect[:, :2] *= -1
  horizontal_theta = np.arctan(ptc_rect[:, 0] / ptc_rect[:, 2])
  horizontal_theta += (ptc_rect[:, 2] < 0) * np.pi
  horizontal_percent = horizontal_theta / (2 * np.pi)
  x = ((horizontal_percent * image.size[0]) + 1880) % image.size[0]
  y = (485.78 * (ptc_rect[:, 1] / ((1 / np.cos(horizontal_theta)) *
       ptc_rect[:, 2]))) + (0.4375 * image.size[1])
  y_inrange = np.logical_and(0 <= y, y < image.size[1])
  rgb = np.array(image)[np.floor(y[y_inrange]).astype(int),
                        np.floor(x[y_inrange]).astype(int)]
  ptc = np.vstack(
      (np.hstack((ptc[y_inrange], rgb)),
       np.hstack((ptc[~y_inrange], np.zeros(((~y_inrange).sum(), 3))))))

  bboxes = []
  for label_3d in labels_3d:
    if label_3d['attributes']['distance'] > DIST_T:
      continue
    if label_3d['attributes']['num_points'] < NUM_PTS_T:
      continue
    rotation_z = (-label_3d['box']['rot_z']
                  if label_3d['box']['rot_z'] < np.pi
                  else 2 * np.pi - label_3d['box']['rot_z'])
    box = np.array(
        (label_3d['box']['cx'], label_3d['box']['cy'],
         label_3d['box']['cz'], label_3d['box']['l'],
         label_3d['box']['w'], label_3d['box']['h'], rotation_z, 0))
    bboxes.append(np.concatenate((box[:3] - box[3:6], box[:3] + box[3:6], box[6:])))
  bboxes = np.vstack(bboxes)
  np.savez_compressed(out_f, pc=ptc, bbox=bboxes)


def main():
  os.mkdir(OUT_D)
  os.mkdir(os.path.join(OUT_D, 'train'))
  file_list_train = get_full_file_list(DATASET_TRAIN_PATH)
  calib_train = get_calibration(DATASET_TRAIN_PATH)
  train_seqs = []
  for seq_name, file_name in tqdm.tqdm(file_list_train):
    train_seq_name = os.path.join('train', f'{seq_name}--{file_name}.npy')
    labels_3d = file_list_train[(seq_name, file_name)]
    train_seqs.append(train_seq_name)
    out_f = os.path.join(OUT_D, train_seq_name)
    process_3d(DATASET_TRAIN_PATH, calib_train, seq_name, file_name, labels_3d, out_f)
  with open(os.path.join(OUT_D, 'train.txt'), 'w') as f:
    f.writelines([l + '\n' for l in train_seqs])

  os.mkdir(os.path.join(OUT_D, 'test'))
  file_list_test = get_full_file_list(DATASET_TEST_PATH)
  calib_test = get_calibration(DATASET_TEST_PATH)
  test_seqs = []
  for seq_name, file_name in tqdm.tqdm(file_list_test):
    test_seq_name = os.path.join('test', f'{seq_name}--{file_name}.npy')
    labels_3d = file_list_test[(seq_name, file_name)]
    test_seqs.append(test_seq_name)
    out_f = os.path.join(OUT_D, test_seq_name)
    process_3d(DATASET_TEST_PATH, calib_test, seq_name, file_name, labels_3d, out_f)
  with open(os.path.join(OUT_D, 'test.txt'), 'w') as f:
    f.writelines([l + '\n' for l in test_seqs])


if __name__ == '__main__':
  main()
