import logging
import os

import numpy as np

from lib.dataset import SparseVoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.utils import read_txt


CLASSES = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa',
    'bookcase', 'board', 'clutter'
]

INSTANCE_SUB_CLASSES = (6, 7, 8, 9, 10, 11)  # door table chair sofa bookcase board
MOVABLE_SUB_CLASSES = (7, 8, 9, 10, 11)  # table chair sofa bookcase board


class Stanford3DDataset(SparseVoxelizationDataset):

  CLIP_BOUND = None
  VOXEL_SIZE = 0.05
  NUM_IN_CHANNEL = 4

  # Augmentation arguments
  # Rotation and elastic distortion distorts thin bounding boxes such as ceiling, wall, or floor.
  ROTATION_AUGMENTATION_BOUND = ((0, 0), (0, 0), (0, 0))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))

  ROTATION_AXIS = 'z'
  LOCFEAT_IDX = 2
  NUM_LABELS = 13
  INSTANCE_LABELS = list(range(13))
  IGNORE_LABELS = None

  DATA_PATH_FILE = {
      DatasetPhase.Train: 'train.txt',
      DatasetPhase.Val: 'val.txt',
      DatasetPhase.TrainVal: 'trainval.txt',
      DatasetPhase.Test: 'test.txt'
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
    data_root = config.stanford3d_path
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

  def load_datafile(self, index):
    pointcloud = np.load(self.data_root / self.data_paths[index])['arr_0']
    return pointcloud, None, None

  def get_instance_mask(self, semantic_labels, instance_labels):
    return instance_labels >= 0


class Stanford3DSubsampleDataset(Stanford3DDataset):

  # Augmentation arguments
  # Turn rotation and elastic distortion augmentation back on.
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36),
                                 (-np.pi / 36, np.pi / 36))
  ELASTIC_DISTORT_PARAMS = ((0.1, 0.2), (0.4, 0.8))

  # Only predict instance labels of our interest.
  IGNORE_LABELS = tuple(set(range(13)) - set(INSTANCE_SUB_CLASSES))
  INSTANCE_LABELS = INSTANCE_SUB_CLASSES

  def get_instance_mask(self, semantic_labels, instance_labels):
    return semantic_labels >= 0


class Stanford3DMovableObjectsDatasets(Stanford3DDataset):

  # Augmentation arguments
  # Turn rotation and elastic distortion augmentation back on.
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36),
                                 (-np.pi / 36, np.pi / 36))
  ELASTIC_DISTORT_PARAMS = ((0.1, 0.2), (0.4, 0.8))

  # Only predict instance labels of our interest.
  IGNORE_LABELS = tuple(set(range(13)) - set(MOVABLE_SUB_CLASSES))
  INSTANCE_LABELS = MOVABLE_SUB_CLASSES

  def get_instance_mask(self, semantic_labels, instance_labels):
    return semantic_labels >= 0


class Stanford3DMovableObjects3cmDatasets(Stanford3DMovableObjectsDatasets):
  VOXEL_SIZE = 0.03
