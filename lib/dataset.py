import random
from abc import ABC
from collections import defaultdict
from pathlib import Path

import numpy as np
from enum import Enum
from torch.utils.data import Dataset, DataLoader

import lib.transforms as t
from lib.dataloader import InfSampler
from lib.pc_utils import read_plyfile, get_bbox
from lib.transforms import cfl_collate_fn_factory, cflt_collate_fn_factory, elastic_distortion
from lib.voxelizer import Voxelizer


class DatasetPhase(Enum):
  Train = 0
  Val = 1
  Val2 = 2
  TrainVal = 3
  Test = 4


def datasetphase_2str(arg):
  if arg == DatasetPhase.Train:
    return 'train'
  elif arg == DatasetPhase.Val:
    return 'val'
  elif arg == DatasetPhase.Val2:
    return 'val2'
  elif arg == DatasetPhase.TrainVal:
    return 'trainval'
  elif arg == DatasetPhase.Test:
    return 'test'
  else:
    raise ValueError('phase must be one of dataset enum.')


def str2datasetphase_type(arg):
  if arg.upper() == 'TRAIN':
    return DatasetPhase.Train
  elif arg.upper() == 'VAL':
    return DatasetPhase.Val
  elif arg.upper() == 'VAL2':
    return DatasetPhase.Val2
  elif arg.upper() == 'TRAINVAL':
    return DatasetPhase.TrainVal
  elif arg.upper() == 'TEST':
    return DatasetPhase.Test
  else:
    raise ValueError('phase must be one of train/val/test')


def cache(func):

  def wrapper(self, *args, **kwargs):
    # Assume that args[0] is index
    index = args[0]
    if self.cache:
      if index not in self.cache_dict[func.__name__]:
        results = func(self, *args, **kwargs)
        self.cache_dict[func.__name__][index] = results
      return self.cache_dict[func.__name__][index]
    else:
      return func(self, *args, **kwargs)

  return wrapper


class DictDataset(Dataset, ABC):

  def __init__(self,
               data_paths,
               input_transform=None,
               target_transform=None,
               cache=False,
               data_root='/'):
    """
    data_paths: list of lists, [[str_path_to_input, str_path_to_label], [...]]
    """
    super().__init__()

    # Allows easier path concatenation
    if not isinstance(data_root, Path):
      data_root = Path(data_root)

    self.data_root = data_root
    self.data_paths = sorted(data_paths)
    self.input_transform = input_transform
    self.target_transform = target_transform

    # dictionary of input
    self.data_loader_dict = {
        'input': (self.load_input, self.input_transform),
        'target': (self.load_target, self.target_transform)
    }

    # For large dataset, do not cache
    self.cache = cache
    self.cache_dict = defaultdict(dict)
    self.loading_key_order = ['input', 'target']

  def load_input(self, index):
    raise NotImplementedError

  def load_target(self, index):
    raise NotImplementedError

  def get_classnames(self):
    pass

  def reorder_result(self, result):
    return result

  def __getitem__(self, index):
    out_array = []
    for k in self.loading_key_order:
      loader, transformer = self.data_loader_dict[k]
      v = loader(index)
      if transformer:
        v = transformer(v)
      out_array.append(v)
    return out_array

  def __len__(self):
    return len(self.data_paths)


class VoxelizationDatasetBase(DictDataset, ABC):
  USE_RGB = True
  IS_TEMPORAL = False
  CLIP_BOUND = None
  ROTATION_AXIS = None
  LOCFEAT_IDX = None
  NUM_IN_CHANNEL = 3
  NUM_LABELS = -1  # Number of labels in the dataset, including all ignore classes
  IGNORE_LABELS = None  # labels that are not evaluated

  # Target bounging box normalization
  BBOX_NORMALIZE_MEAN = np.array((0., 0., 0., 0., 0., 0.))
  BBOX_NORMALIZE_STD = np.array((1., 1., 1., 1., 1., 1.))

  def __init__(self,
               data_paths,
               input_transform=None,
               target_transform=None,
               cache=False,
               data_root='/',
               explicit_rotation=-1,
               ignore_mask=255,
               return_transformation=False,
               **kwargs):
    """
    ignore_mask: label value for ignore class. It will not be used as a class in the loss or
                 evaluation.
    explicit_rotation: # of discretization of 360 degree. # data would be
                       num_data * explicit_rotation
    """
    super().__init__(
        data_paths,
        input_transform=input_transform,
        target_transform=target_transform,
        cache=cache,
        data_root=data_root)

    self.ignore_mask = ignore_mask
    self.explicit_rotation = explicit_rotation
    self.return_transformation = return_transformation

  def _augment_locfeat(self, coords, feats):
    if self.LOCFEAT_IDX is not None:
      height = coords[:, self.LOCFEAT_IDX].copy()
      height -= np.percentile(height, 0.99)
      return np.hstack((feats, height[:, None]))
    return feats

  def get_instance_mask(self, semantic_labels, instance_labels):
    raise NotImplementedError

  def __getitem__(self, index):
    raise NotImplementedError

  def load_datafile(self, index):
    filepath = self.data_root / self.data_paths[index]
    return read_plyfile(filepath), None, None

  def test_pointcloud(self, pred_dir):
    raise NotImplementedError

  def __len__(self):
    num_data = len(self.data_paths)
    if self.explicit_rotation > 1:
      return num_data * self.explicit_rotation
    return num_data


class SparseVoxelizationDataset(VoxelizationDatasetBase):
  """This dataset loads RGB point clouds and their labels as a list of points
  and voxelizes the pointcloud with sufficient data augmentation.
  """

  IS_ROTATION_BBOX = False
  HAS_GT_BBOX = False

  # Voxelization arguments
  VOXEL_SIZE = 0.05  # 5cm

  # Augmentation arguments
  SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 6, np.pi / 6), (-np.pi, np.pi), (-np.pi / 6, np.pi / 6))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.05, 0.05), (-0.2, 0.2))
  ELASTIC_DISTORT_PARAMS = None

  def __init__(self,
               data_paths,
               input_transform=None,
               target_transform=None,
               data_root='/',
               explicit_rotation=-1,
               ignore_label=255,
               return_transformation=False,
               augment_data=False,
               config=None,
               **kwargs):
    if explicit_rotation > 0:
      raise NotImplementedError

    self.augment_data = augment_data
    self.config = config
    super().__init__(
        data_paths,
        input_transform=input_transform,
        target_transform=target_transform,
        cache=cache,
        data_root=data_root,
        explicit_rotation=explicit_rotation,
        ignore_mask=ignore_label,
        return_transformation=return_transformation)

    self.sparse_voxelizer = Voxelizer(
        voxel_size=self.VOXEL_SIZE,
        clip_bound=self.CLIP_BOUND,
        use_augmentation=augment_data,
        scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
        rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
        translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND,
        ignore_label=ignore_label)

    # map labels not evaluated to ignore_label
    if self.IGNORE_LABELS is not None:
      label_map = {}
      n_used = 0
      for l in range(self.NUM_LABELS):
        if l in self.IGNORE_LABELS:
          label_map[l] = self.ignore_mask
        elif l not in self.INSTANCE_LABELS:
          label_map[l] = n_used
          n_used += 1
      for l in self.INSTANCE_LABELS:
        label_map[l] = n_used
        n_used += 1
      label_map[self.ignore_mask] = self.ignore_mask
      self.label_map = label_map
      self.NUM_LABELS -= len(self.IGNORE_LABELS)

  def convert_mat2cfl(self, mat):
    # Generally, xyz, rgb, label
    return mat[:, :3], mat[:, 3:-2], mat[:, -2:]

  def _augment_elastic_distortion(self, pointcloud):
    if self.ELASTIC_DISTORT_PARAMS is not None:
      if random.random() < 0.95:
        for granularity, magnitude in self.ELASTIC_DISTORT_PARAMS:
          pointcloud = elastic_distortion(pointcloud, granularity, magnitude)
    return pointcloud

  def __getitem__(self, index):
    pointcloud, bboxes, center = self.load_datafile(index)
    if self.augment_data and self.config.elastic_distortion:
      pointcloud = self._augment_elastic_distortion(pointcloud)
    coords, feats, labels = self.convert_mat2cfl(pointcloud)
    feats = self._augment_locfeat(coords, feats)
    coords, feats, labels, transformation = self.sparse_voxelizer.voxelize(
        coords, feats, labels, center=center)

    # transform bboxes to match the voxelization transformation.
    # TODO(jgwak): Support rotation augmentation.
    if self.HAS_GT_BBOX:
      bboxes_xyz = np.hstack((bboxes[:, :6].reshape(-1, 3), np.ones((bboxes.shape[0] * 2, 1))))
      bboxes_xyz = bboxes_xyz @ transformation.reshape(4, 4).T
      bboxes_xyz = (bboxes_xyz[:, :3] / bboxes_xyz[:, 3, None]).reshape(-1, 2, 3)
      bboxes_xyz = np.hstack((bboxes_xyz.min(1), bboxes_xyz.max(1)))
      bboxes = np.hstack((bboxes_xyz, bboxes[:, 6:]))

    if labels is None:
      semantic_labels = np.ones(coords.shape[0]) * self.config.ignore_label
      instance_labels = np.ones(coords.shape[0]) * self.config.ignore_label
    else:
      semantic_labels, instance_labels = labels.T

    # map labels not used for evaluation to ignore_label
    if self.input_transform is not None:
      coords, feats, bboxes = self.input_transform(coords, feats, bboxes)
    if self.target_transform is not None:
      coords, feats, bboxes = self.target_transform(coords, feats, bboxes)
    if self.IGNORE_LABELS is not None:
      semantic_labels = np.array([self.label_map[x] for x in semantic_labels], dtype=np.int)
      if self.HAS_GT_BBOX:
        bboxes[:, -1] = np.array([self.label_map[int(x)] for x in bboxes[:, -1]])

    # Normalize rotation.
    if self.augment_data and self.HAS_GT_BBOX and self.IS_ROTATION_BBOX:
      if self.config.normalize_rotation:
        rot_bin = np.floor((bboxes[:, 6] + np.pi / 4) / (np.pi / 2))
        idxs = list(range(3))
        idxs.remove(self.LOCFEAT_IDX)
        assert len(idxs) == 2
        bbox_size = (bboxes[:, 3:6] - bboxes[:, :3]) / 2
        bbox_center = (bboxes[:, 3:6] + bboxes[:, :3]) / 2
        bboxes[:, 6] = bboxes[:, 6] - rot_bin * np.pi / 2
        is_rotated = np.where(rot_bin % 2)[0]
        bbox_size[np.ix_(is_rotated, idxs)] = bbox_size[np.ix_(is_rotated, list(reversed(idxs)))]
        bboxes = np.hstack((bbox_center - bbox_size, bbox_center + bbox_size, bboxes[:, 6:]))
      elif self.config.normalize_rotation2:
        bboxes[:, 6] = (bboxes[:, 6] % np.pi - np.pi / 2) * 2

    if not self.HAS_GT_BBOX:
      instance_mask = self.get_instance_mask(semantic_labels, instance_labels)
      bboxes, instance_labels = get_bbox(
          coords, semantic_labels, instance_labels, instance_mask, self.ignore_mask)
    labels = np.hstack((np.expand_dims(semantic_labels, 1), np.expand_dims(instance_labels, 1)))

    return_args = [coords, feats, labels, bboxes]
    if self.return_transformation:
      transformation = np.expand_dims(transformation, 0)
      return_args.extend([pointcloud.astype(np.float32), transformation.astype(np.float32)])
    return tuple(return_args)


def initialize_data_loader(DatasetClass,
                           config,
                           phase,
                           threads,
                           shuffle,
                           repeat,
                           augment_data,
                           batch_size,
                           limit_numpoints,
                           input_transform=None,
                           target_transform=None):
  if isinstance(phase, str):
    phase = str2datasetphase_type(phase)

  if config.return_transformation:
    collate_fn = cflt_collate_fn_factory(DatasetClass.IS_ROTATION_BBOX, limit_numpoints, config)
  else:
    collate_fn = cfl_collate_fn_factory(DatasetClass.IS_ROTATION_BBOX, limit_numpoints, config)

  input_transforms = []
  if input_transform is not None:
    input_transforms += input_transform

  if augment_data:
    input_transforms += [
        t.RandomHorizontalFlip(DatasetClass.ROTATION_AXIS, DatasetClass.IS_TEMPORAL),
        t.HeightTranslation(config.data_aug_height_trans_std),
        t.HeightJitter(config.data_aug_height_jitter_std),
    ]
    if DatasetClass.USE_RGB:
      input_transforms += [
          t.ChromaticTranslation(config.data_aug_color_trans_ratio),
          t.ChromaticJitter(config.data_aug_color_jitter_std),
      ]

  if len(input_transforms) > 0:
    input_transforms = t.Compose(input_transforms)
  else:
    input_transforms = None

  dataset = DatasetClass(
      config,
      input_transform=input_transforms,
      target_transform=target_transform,
      cache=config.cache_data,
      augment_data=augment_data,
      phase=phase)

  if repeat:
    # Use the inf random sampler
    data_loader = DataLoader(
        dataset=dataset,
        num_workers=threads,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=InfSampler(dataset, shuffle))
  else:
    # Default shuffle=False
    data_loader = DataLoader(
        dataset=dataset,
        num_workers=threads,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle)

  return data_loader
