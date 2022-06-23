import logging
import os

import numpy as np

from lib.dataset import SparseVoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.utils import read_txt


CLASS_LABELS = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser', 'night_stand',
                'bookshelf', 'bathtub')


class SUNRGBDDataset(SparseVoxelizationDataset):

  IS_ROTATION_BBOX = True
  HAS_GT_BBOX = True

  # Voxelization arguments
  CLIP_BOUND = None
  VOXEL_SIZE = 0.05
  NUM_IN_CHANNEL = 4

  # Augmentation arguments
  # TODO(jgwak): Support rotation augmentation.
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64),
                                 (-np.pi / 64, np.pi / 64))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
  ELASTIC_DISTORT_PARAMS = ((0.1, 0.2), (0.4, 0.8))

  ROTATION_AXIS = 'z'
  LOCFEAT_IDX = 2
  NUM_LABELS = 10
  INSTANCE_LABELS = list(range(10))

  DATA_PATH_FILE = {
      DatasetPhase.Train: 'train.txt',
      DatasetPhase.TrainVal: 'trainval.txt',
      DatasetPhase.Val: 'val.txt',
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
    data_root = config.sunrgbd_path
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
    datum = np.load(self.data_root / self.data_paths[index])
    pointcloud, bboxes = datum['pc'], datum['bbox']
    pointcloud[:, 3:] *= 255
    centers = (bboxes[:, 3:6] + bboxes[:, :3]) / 2
    sizes = np.maximum(np.abs(bboxes[:, 3:6] - bboxes[:, :3]), 0.01)
    bboxes = np.hstack((centers - sizes / 2, centers + sizes / 2, bboxes[:, 6:]))
    bboxes[:, 6] *= -1
    return pointcloud, bboxes, None

  def convert_mat2cfl(self, mat):
    # Generally, xyz, rgb, label
    return mat[:, :3], mat[:, 3:], None
