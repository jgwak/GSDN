import os
import imageio
import pickle
import numpy as np
import logging

from lib.dataset import DictDataset, SparseVoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.utils import read_txt


class Synthia2dDataset(DictDataset):
  NUM_LABELS = 16

  def __init__(self, data_path_file, input_transform=None, target_transform=None):
    with open(data_path_file, 'r') as f:
      data_paths = pickle.load(f)
    super(SynthiaDataset, self).__init__(data_paths, input_transform, target_transform)

  @staticmethod
  def load_extrinsics(extrinsics_file):
    """Load the camera extrinsics from a .txt file.
    """
    lines = read_txt(extrinsics_file)
    params = [float(x) for x in lines[0].split(' ')]
    extrinsics_matrix = np.asarray(params).reshape([4, 4])
    return extrinsics_matrix

  @staticmethod
  def load_intrinsics(intrinsics_file):
    """Load the camera intrinsics from a intrinsics.txt file.

    intrinsics.txt: a text file containing 4 values that represent (in this order) {focal length,
                    principal-point-x, principal-point-y, baseline (m) with the corresponding right
                    camera}
    """
    lines = read_txt(intrinsics_file)
    assert len(lines) == 7
    intrinsics = {
        'focal_length': float(lines[0]),
        'pp_x': float(lines[2]),
        'pp_y': float(lines[4]),
        'baseline': float(lines[6]),
    }
    return intrinsics

  @staticmethod
  def load_depth(depth_file):
    """Read a single depth map (.png) file.

    1280x760
    760 rows, 1280 columns.
    Depth is encoded in any of the 3 channels in centimetres as an ushort.
    """
    img = np.asarray(imageio.imread(depth_file, format='PNG-FI'))  # uint16
    img = img.astype(np.int32)  # Convert to int32 for torch compatibility
    return img

  @staticmethod
  def load_label(label_file):
    """Load the ground truth semantic segmentation label.

    Annotations are given in two channels. The first channel contains the class of that pixel
    (see the table below). The second channel contains the unique ID of the instance for those
    objects that are dynamic (cars, pedestrians, etc.).

    Class         R       G       B       ID

    Void          0       0       0       0
    Sky             128   128     128     1
    Building        128   0       0       2
    Road            128   64      128     3
    Sidewalk        0     0       192     4
    Fence           64    64      128     5
    Vegetation      128   128     0       6
    Pole            192   192     128     7
    Car             64    0       128     8
    Traffic Sign    192   128     128     9
    Pedestrian      64    64      0       10
    Bicycle         0     128     192     11
    Lanemarking   0       172     0       12
    Reserved      -       -       -       13
    Reserved      -       -       -       14
    Traffic Light 0       128     128     15
    """
    img = np.asarray(imageio.imread(label_file, format='PNG-FI'))  # uint16
    img = img.astype(np.int32)  # Convert to int32 for torch compatibility
    return img

  @staticmethod
  def load_rgb(rgb_file):
    """Load RGB images. 1280x760 RGB images used for training.

    760 rows, 1280 columns.
    """
    img = np.array(imageio.imread(rgb_file))  # uint8
    return img


class SynthiaDataset(SparseVoxelizationDataset):

  # Voxelization arguments
  CLIP_BOUND = ((-2000, 2000), (-2000, 2000), (-2000, 2000))
  VOXEL_SIZE = 30
  NUM_IN_CHANNEL = 4

  # Target bounging box normalization
  BBOX_NORMALIZE_MEAN = np.array((0., 0., 0., 10.802, 6.258, 10.543))
  BBOX_NORMALIZE_STD = np.array((3.331, 1.507, 3.007, 5.179, 1.177, 4.268))

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi, np.pi), (-np.pi / 64,
                                                                              np.pi / 64))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (0, 0), (-0.2, 0.2))

  ROTATION_AXIS = 'y'
  LOCFEAT_IDX = 1
  NUM_LABELS = 16
  INSTANCE_LABELS = (8, 10, 11)
  IGNORE_LABELS = (0, 1, 13, 14)  # void, sky, reserved, reserved

  DATA_PATH_FILE = {
      DatasetPhase.Train: 'train.txt',
      DatasetPhase.Val: 'val.txt',
      DatasetPhase.Val2: 'val2.txt',
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
    data_root = config.synthia_path
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
    pointcloud, bboxes, _ = super().load_datafile(index)
    return pointcloud, bboxes, np.zeros(3)

  def get_instance_mask(self, semantic_labels, instance_labels):
    return instance_labels != 0
