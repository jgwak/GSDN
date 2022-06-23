import logging
import os

import numpy as np

from lib.dataset import SparseVoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.pc_utils import read_plyfile
from lib.utils import read_txt

# https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py#L57
CLASS_LABELS = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet',
                'sink', 'bathtub', 'otherfurniture')
VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}


class ScannetDataset(SparseVoxelizationDataset):

  # Voxelization arguments
  CLIP_BOUND = None
  VOXEL_SIZE = 0.05
  NUM_IN_CHANNEL = 4

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64),
                                 (-np.pi, np.pi))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))

  ROTATION_AXIS = 'z'
  LOCFEAT_IDX = 2
  NUM_LABELS = 41  # Will be converted to 18 as defined in IGNORE_LABELS.
  INSTANCE_LABELS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
  IGNORE_LABELS = tuple(set(range(41)) - set(VALID_CLASS_IDS))

  DATA_PATH_FILE = {
      DatasetPhase.Train: 'scannetv2_train.txt',
      DatasetPhase.TrainVal: 'scannetv2_trainval.txt',
      DatasetPhase.Val: 'scannetv2_val.txt',
      DatasetPhase.Test: 'scannetv2_test.txt'
  }

  def __init__(self,
               config,
               input_transform=None,
               target_transform=None,
               augment_data=True,
               cache=False,
               phase=DatasetPhase.Train):
    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)
    data_root = config.scannet_path
    data_paths = read_txt(os.path.join(data_root, self.DATA_PATH_FILE[phase]))
    logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
    super().__init__(
        data_paths,
        data_root=data_root,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=config.ignore_label,
        return_transformation=config.return_transformation,
        augment_data=augment_data,
        config=config)

  def get_instance_mask(self, semantic_labels, instance_labels):
    return self.NUM_LABELS - len(self.INSTANCE_LABELS) <= semantic_labels


class Scannet3cmDataset(ScannetDataset):
  VOXEL_SIZE = 0.03


class ScannetAlignedDataset(ScannetDataset):

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36),
                                 (-np.pi / 36, np.pi / 36))
  ELASTIC_DISTORT_PARAMS = ((0.1, 0.2), (0.4, 0.8))
  INSTANCE_LABELS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
  IGNORE_LABELS = tuple(set(range(41)) - set(INSTANCE_LABELS))

  def load_datafile(self, index):
    filepath = self.data_root / self.data_paths[index]
    scene_id = os.path.splitext(self.data_paths[index].split(os.sep)[-1])[0]
    scene_f = self.config.scannet_alignment_path % (scene_id, scene_id)
    ptc = read_plyfile(filepath)
    if os.path.isfile(scene_f):
      alignment_txt = [l for l in read_txt(scene_f) if l.startswith('axisAlignment = ')][0]
      rot = np.array([float(x) for x in alignment_txt[16:].split()]).reshape(4, 4)
      xyz1 = np.hstack((ptc[:, :3], np.ones((ptc.shape[0], 1))))
      ptc[:, :3] = (xyz1 @ rot.T)[:, :3]
    return ptc, None, None

  def get_instance_mask(self, semantic_labels, instance_labels):
    return semantic_labels >= 0


class ScannetVoteNetDataset(SparseVoxelizationDataset):

  USE_RGB = False
  CLIP_BOUND = None
  VOXEL_SIZE = 0.05
  NUM_IN_CHANNEL = 1

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36),
                                 (-np.pi / 36, np.pi / 36))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
  ELASTIC_DISTORT_PARAMS = ((0.1, 0.2), (0.4, 0.8))

  ROTATION_AXIS = 'z'
  LOCFEAT_IDX = 2
  NUM_LABELS = 41  # Will be converted to 18 as defined in IGNORE_LABELS.
  INSTANCE_LABELS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
  IGNORE_LABELS = tuple(set(range(41)) - set(INSTANCE_LABELS))

  DATA_PATH_FILE = {
      DatasetPhase.Train: 'scannet_votenet_train.txt',
      DatasetPhase.Val: 'scannet_votenet_val.txt',
      DatasetPhase.TrainVal: 'scannet_votenet_trainval.txt',
      DatasetPhase.Test: 'scannet_votenet_test.txt'
  }

  def __init__(self,
               config,
               input_transform=None,
               target_transform=None,
               augment_data=True,
               cache=False,
               phase=DatasetPhase.Train):
    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)
    data_root = config.scannet_votenet_path
    data_paths = read_txt(os.path.join(data_root, self.DATA_PATH_FILE[phase]))
    logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
    super().__init__(
        data_paths,
        data_root=data_root,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=config.ignore_label,
        return_transformation=config.return_transformation,
        augment_data=augment_data,
        config=config)

  def get_instance_mask(self, semantic_labels, instance_labels):
    return semantic_labels >= 0


class ScannetVoteNet3cmDataset(ScannetVoteNetDataset):
  VOXEL_SIZE = 0.03


class ScannetVoteNetRGBDataset(SparseVoxelizationDataset):

  CLIP_BOUND = None
  VOXEL_SIZE = 0.05
  NUM_IN_CHANNEL = 4

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36),
                                 (-np.pi / 36, np.pi / 36))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
  ELASTIC_DISTORT_PARAMS = ((0.1, 0.2), (0.4, 0.8))

  ROTATION_AXIS = 'z'
  LOCFEAT_IDX = 2
  NUM_LABELS = 41  # Will be converted to 18 as defined in IGNORE_LABELS.
  INSTANCE_LABELS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
  IGNORE_LABELS = tuple(set(range(41)) - set(INSTANCE_LABELS))

  DATA_PATH_FILE = {
      DatasetPhase.Train: 'scannet_votenet_train.txt',
      DatasetPhase.Val: 'scannet_votenet_val.txt',
      DatasetPhase.TrainVal: 'scannet_votenet_trainval.txt',
      DatasetPhase.Test: 'scannet_votenet_test.txt'
  }

  def __init__(self,
               config,
               input_transform=None,
               target_transform=None,
               augment_data=True,
               cache=False,
               phase=DatasetPhase.Train):
    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)
    data_root = config.scannet_votenetrgb_path
    data_paths = read_txt(os.path.join(data_root, self.DATA_PATH_FILE[phase]))
    logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
    super().__init__(
        data_paths,
        data_root=data_root,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=config.ignore_label,
        return_transformation=config.return_transformation,
        augment_data=augment_data,
        config=config)

  def get_instance_mask(self, semantic_labels, instance_labels):
    return semantic_labels >= 0


class ScannetVoteNetRGB3cmDataset(ScannetVoteNetRGBDataset):
  VOXEL_SIZE = 0.03


class ScannetVoteNetRGB25mmDataset(ScannetVoteNetRGBDataset):
  VOXEL_SIZE = 0.025
