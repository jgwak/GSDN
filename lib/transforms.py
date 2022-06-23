import itertools
import random

import logging
import numpy as np
import scipy
import scipy.ndimage
import torch
import lib.detection_utils as detection_utils

import MinkowskiEngine as ME


class RandomHorizontalFlip(object):

  def __init__(self, upright_axis, is_temporal):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.is_temporal = is_temporal
    self.D = 4 if is_temporal else 3
    if upright_axis is None:
      self.upright_axis = None
    else:
      self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
      # Use the rest of axes for flipping.
      self.horz_axes = set(range(self.D)) - set([self.upright_axis])

  def __call__(self, coords, feats, labels):
    if self.upright_axis is not None:
      if random.random() < 0.95:
        for curr_ax in self.horz_axes:
          if random.random() < 0.5:
            coord_max = np.max(coords[:, curr_ax])
            coords[:, curr_ax] = coord_max - coords[:, curr_ax]
            if labels is not None:
              label_min = coord_max - labels[:, 3 + curr_ax]
              label_max = coord_max - labels[:, curr_ax]
              labels[:, curr_ax] = label_min
              labels[:, 3 + curr_ax] = label_max
              if labels.shape[1] == 8:
                labels[:, 6] = ((3 - curr_ax) * np.pi - labels[:, 6]) % (2 * np.pi) - np.pi
    return coords, feats, labels


class ChromaticTranslation(object):
  """Add random color to the image, input must be an array in [0,255] or a PIL image"""

  def __init__(self, trans_range_ratio=1e-1):
    """
    trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
    """
    self.trans_range_ratio = trans_range_ratio

  def __call__(self, coords, feats, labels):
    if random.random() < 0.95:
      tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
      feats[:, :3] += tr
    return coords, feats, labels


class ChromaticJitter(object):

  def __init__(self, std=0.01):
    self.std = std

  def __call__(self, coords, feats, labels):
    if random.random() < 0.95:
      noise = np.random.randn(feats.shape[0], 3)
      noise *= self.std * 255
      feats[:, :3] += noise
    return coords, feats, labels


class HeightTranslation(object):

  def __init__(self, std=0.01):
    self.std = std

  def __call__(self, coords, feats, labels):
    if feats.shape[1] > 3 and random.random() < 0.95:
      feats[:, -1] += np.random.randn(1) * self.std
    return coords, feats, labels


class HeightJitter(object):

  def __init__(self, std):
    self.std = std

  def __call__(self, coords, feats, labels):
    if feats.shape[1] > 3 and random.random() < 0.95:
      feats[:, -1] += np.random.randn(feats.shape[0]) * self.std
    return coords, feats, labels


class NormalJitter(object):

  def __init__(self, std):
    self.std = std

  def __call__(self, coords, feats, labels):
    # normal jitter
    if feats.shape[1] > 6 and random.random() < 0.95:
      feats[:, 3:6] += np.random.randn(feats.shape[0], 3) * self.std
    return coords, feats, labels


class Compose(object):
  """Composes several transforms together."""

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, *args):
    for t in self.transforms:
      args = t(*args)
    return args


class cfl_collate_fn_factory:
  """Generates collate function for coords, feats, labels.

    Args:
      limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                       size so that the number of input coordinates is below limit_numpoints.
  """

  def __init__(self, is_rotation_bbox, limit_numpoints, config):
    self.is_rotation_bbox = is_rotation_bbox
    self.limit_numpoints = limit_numpoints
    self.config = config
    self.anchor_ratios = np.reshape(config.rpn_anchor_ratios, (-1, 3))

  def _get_rpn_target(self, gt_boxes, gt_rotations, gt_classes, anchors):
    rpn_match = np.zeros((anchors.shape[0]), dtype=np.int8)
    train_full_anchor = (self.config.load_sparse_gt_data
                         or self.config.rpn_train_anchors_per_batch <= 0)
    num_train_anchors = (anchors.shape[0] if train_full_anchor
                         else self.config.rpn_train_anchors_per_batch)
    rpn_cls = np.ones((anchors.shape[0]), dtype=np.int8) * -1
    rpn_bbox = np.zeros((num_train_anchors, anchors.shape[1]), dtype=np.float32)
    if gt_rotations is None:
      rpn_rotation = None
    else:
      rpn_rotation = np.zeros((num_train_anchors), dtype=np.float32)
    if gt_boxes.size == 0:
      rpn_match[:] = -1
    else:
      gt_overlap_rotation = None
      if self.config.rpn_rotation_overlap and gt_rotations is not None:
        gt_overlap_rotation = gt_rotations
      overlaps = detection_utils.compute_overlaps(
          anchors, gt_boxes, rotations2=gt_overlap_rotation).numpy()
      anchor_iou_argmax = np.argmax(overlaps, axis=1)
      anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
      rpn_match[anchor_iou_max < self.config.rpn_match_negative_iou_threshold] = -1
      nonzero_argmin = np.argwhere(overlaps == np.maximum(np.max(overlaps, axis=0), 1e-5))[:, 0]
      rpn_match[nonzero_argmin] = 1
      rpn_match[anchor_iou_max >= self.config.rpn_match_positive_iou_threshold] = 1
    if not train_full_anchor:
      ids = np.where(rpn_match == 1)[0]
      extra = len(ids) - (self.config.rpn_train_anchors_per_batch // 2)
      if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
      ids = np.where(rpn_match == -1)[0]
      extra = len(ids) - (self.config.rpn_train_anchors_per_batch
                          - np.sum(rpn_match == 1))
      if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    if gt_boxes.size > 0:
      ids = np.where(rpn_match == 1)[0]
      anchor_ids = anchor_iou_argmax[ids]
      rpn_target = detection_utils.get_bbox_target(
          torch.from_numpy(anchors[ids]),
          torch.from_numpy(gt_boxes[anchor_ids]), self.config.rpn_bbox_std)
      rpn_bbox[:len(ids)] = rpn_target.numpy()
      rpn_cls[:len(ids)] = gt_classes[anchor_ids]
      if gt_rotations is not None:
        rpn_rotation[:len(ids)] = gt_rotations[anchor_ids]
    return rpn_match, rpn_bbox, rpn_rotation, rpn_cls

  def _get_rpn_targets(self, anchors, gt_boxes, gt_rotations, gt_classes, anchor_masks=None):
    """Given anchors and ground-truth boxes, pre-compute the rpn training target."""
    assert len(gt_boxes) > 0 and gt_boxes[0].shape[1] == anchors.shape[1]
    batch_size = len(gt_boxes)
    # TODO(jgwak): Remove unnecessary zero padding.
    rpn_match = []
    rpn_bbox = []
    rpn_rotation = None if gt_rotations is None else []
    rpn_cls = []
    for batch_idx in range(batch_size):
      gt_rotation = None if gt_rotations is None else gt_rotations[batch_idx]
      batch_anchors = anchors if anchor_masks is None else anchors[anchor_masks[batch_idx]]
      batch_rpn_match, batch_rpn_bbox, batch_rpn_rotation, batch_rpn_cls = self._get_rpn_target(
          gt_boxes[batch_idx], gt_rotation, gt_classes[batch_idx], batch_anchors)
      rpn_match.append(batch_rpn_match)
      rpn_bbox.append(batch_rpn_bbox)
      rpn_cls.append(batch_rpn_cls)
      if gt_rotations is not None:
        rpn_rotation.append(batch_rpn_rotation)
    if anchor_masks is None:
      rpn_match = np.stack(rpn_match)
      rpn_bbox = np.stack(rpn_bbox)
      rpn_cls = np.stack(rpn_cls)
      if gt_rotations is not None:
        rpn_rotation = np.stack(rpn_rotation)
    return rpn_match, rpn_bbox, rpn_rotation, rpn_cls

  def _get_sparse_rpn_targets(self, coords, anchors, anchor_coords, gt_boxes, gt_rotations,
                              gt_classes, rpn_strides):
    def _split_layer(target, num_layer_anchors):
      assert num_layer_anchors[-1] == target.shape[0]
      return np.split(target, num_layer_anchors)
    num_layer_anchors = np.cumsum([x.shape[0] for x in anchor_coords])
    # Precompute anchors that may potentially hit based on input coordinates.
    anchor_dcoords = np.floor(np.concatenate(anchor_coords) / rpn_strides[-1]).astype(int)
    anchor_ddim = anchor_dcoords.max(0) - anchor_dcoords.min(0) + 2
    anchor_idxs = np.ravel_multi_index((anchor_dcoords - anchor_dcoords.min(0)).T, anchor_ddim)
    anchor_kernel_perm = np.array(list(itertools.product(*[range(-1, 2)] * 3)))
    anchor_masks = []
    for coord in coords:
      anchor_hit = np.full(np.prod(anchor_ddim), False)
      coords_dcoords = np.unique(np.floor(coord / rpn_strides[-1]).astype(int), axis=0)
      coords_perm = (np.tile(np.expand_dims(coords_dcoords, 0), (3 ** 3, 1, 1))
                     + np.expand_dims(anchor_kernel_perm, 1)).reshape(-1, 3)
      coords_idxs = np.ravel_multi_index((coords_perm - anchor_dcoords.min(0)).T, anchor_ddim)
      anchor_hit[coords_idxs] = True
      anchor_masks.append(anchor_hit[anchor_idxs])
    rpn_match, rpn_bbox, rpn_rotation, rpn_cls = self._get_rpn_targets(
        anchors, gt_boxes, gt_rotations, gt_classes, anchor_masks=anchor_masks)
    # Split dense regression targets based on layer index.
    anchors = _split_layer(anchors, num_layer_anchors)
    num_anchors = self.anchor_ratios.shape[0]
    num_batch = len(rpn_match)
    anchors_layer = []
    anchor_coords_layer = []
    rpn_match_layer = []
    rpn_bbox_layer = []
    rpn_cls_layer = []
    rpn_rot_layer = []
    for batch_idx in range(num_batch):
      mask_layer = _split_layer(anchor_masks[batch_idx], num_layer_anchors)
      batch_layer_anchors = np.cumsum([i.sum() for i in mask_layer][:-1])
      rpn_positive_mask = np.where(rpn_match[batch_idx] == 1)
      rpn_aligned_positive_mask = np.where(~np.all(rpn_bbox[batch_idx] == 0, -1))

      anchors_layer.append([ac[ml] for ac, ml in zip(anchors, mask_layer)])
      anchor_coords_layer.append([ac[ml] for ac, ml in zip(anchor_coords, mask_layer)])
      rpn_match_layer.append(_split_layer(rpn_match[batch_idx], batch_layer_anchors))

      rpn_bbox_aligned = np.zeros_like(rpn_bbox[batch_idx])
      rpn_bbox_aligned[rpn_positive_mask] = rpn_bbox[batch_idx][rpn_aligned_positive_mask]
      rpn_bbox_aligned = _split_layer(rpn_bbox_aligned, batch_layer_anchors)
      rpn_bbox_layer.append(rpn_bbox_aligned)

      rpn_cls_aligned = np.ones_like(rpn_cls[batch_idx]) * -1
      rpn_cls_aligned[rpn_positive_mask] = rpn_cls[batch_idx][rpn_aligned_positive_mask]
      rpn_cls_aligned = _split_layer(rpn_cls_aligned, batch_layer_anchors)
      rpn_cls_layer.append(rpn_cls_aligned)

      if rpn_rotation is not None:
        rpn_rotation_aligned = np.zeros_like(rpn_rotation[batch_idx])
        rpn_rotation_aligned[rpn_positive_mask] = rpn_rotation[batch_idx][rpn_aligned_positive_mask]
        rpn_rotation_aligned = _split_layer(rpn_rotation_aligned, batch_layer_anchors)
        rpn_rot_layer.append(rpn_rotation_aligned)
    anchors_layer = list(zip(*anchors_layer))
    anchor_coords_layer = list(zip(*anchor_coords_layer))
    rpn_match_layer = list(zip(*rpn_match_layer))
    rpn_bbox_layer = list(zip(*rpn_bbox_layer))
    rpn_cls_layer = list(zip(*rpn_cls_layer))
    rpn_rot_layer = None if rpn_rotation is None else list(zip(*rpn_rot_layer))
    # Accumulate regression target in sparse tensor format.
    sparse_anchor_centers = []
    sparse_anchor_coords = []
    sparse_rpn_matches = []
    sparse_rpn_bboxes = []
    sparse_rpn_cls = []
    sparse_rpn_rotations = [] if rpn_rotation is not None else None
    for layer_idx in range(len(anchor_coords)):
      tensor_stride = rpn_strides[layer_idx]
      sparse_anchor_centers.append(
          torch.from_numpy(np.vstack(anchors_layer[layer_idx]).reshape(-1, num_anchors * 6)))
      sub_anchor_coords = ME.utils.batched_coordinates([ac[::num_anchors] for ac
                                                        in anchor_coords_layer[layer_idx]])
      sub_anchor_coords[:, 1:] -= tensor_stride // 2
      sparse_anchor_coords.append(sub_anchor_coords)
      sparse_rpn_matches.append(
          torch.from_numpy(np.concatenate(rpn_match_layer[layer_idx]).reshape(-1, num_anchors)))
      sparse_rpn_bboxes.append(
          torch.from_numpy(np.concatenate(rpn_bbox_layer[layer_idx]).reshape(-1, num_anchors * 6)))
      sparse_rpn_cls.append(
          torch.from_numpy(np.concatenate(rpn_cls_layer[layer_idx]).reshape(-1, num_anchors)))
      if sparse_rpn_rotations is not None:
        sparse_rpn_rotations.append(
            torch.from_numpy(np.concatenate(rpn_rot_layer[layer_idx]).reshape(-1, num_anchors)))
    # Precompute positive/ambiguous target coordinates for sparse generative RPN.
    anchor_match_coords = []
    for batch_idx in range(num_batch):
      batch_match_coords = []
      for layer_idx in range(len(anchor_coords)):
        layer_match = rpn_match_layer[layer_idx][batch_idx]
        layer_coords = anchor_coords_layer[layer_idx][batch_idx]
        positive_match = np.where(layer_match == 1)
        if np.any(positive_match):
          positive_coords = np.unique(layer_coords[positive_match], axis=0)
        else:
          positive_coords = np.zeros((0, 3))
        ambiguous_match = np.where(layer_match == 0)
        if np.any(ambiguous_match):
          ambiguous_coords = np.unique(layer_coords[ambiguous_match], axis=0)
        else:
          ambiguous_coords = np.zeros((0, 3))
        batch_match_coords.append((positive_coords, ambiguous_coords))
      anchor_match_coords.append(batch_match_coords)
    return (anchor_match_coords, sparse_anchor_centers, sparse_anchor_coords, sparse_rpn_matches,
            sparse_rpn_bboxes, sparse_rpn_rotations, sparse_rpn_cls)

  def _get_backbone_shapes(self, coords):
    """Get backbone network output shapes based on input coordinates."""
    def _round_down(x, d):
      return (torch.floor(x.float() / d) * d).int()
    final_pixel_dist = self.config.rpn_strides[-1]
    assert torch.all(_round_down(coords[:, 1:].min(0)[0], final_pixel_dist) == 0)
    output_max = _round_down(coords[:, 1:].max(0)[0], final_pixel_dist)
    voxel_shape = (output_max + final_pixel_dist).numpy()
    backbone_shapes = np.array([voxel_shape // d for d in self.config.rpn_strides])
    return backbone_shapes

  def __call__(self, list_data):
    coords, feats, point_labels, bboxes = list(zip(*list_data))
    coords_batch, feats_batch, labels_batch = [], [], []
    bboxes_coords, bboxes_rotations, bboxes_cls = [], [], []

    batch_num_points = 0
    last_batch_id = 0
    for batch_id in range(len(list_data)):
      num_points = coords[batch_id].shape[0]
      batch_num_points += num_points
      if self.limit_numpoints and batch_num_points > self.limit_numpoints:
        num_full_points = sum(len(c) for c in coords)
        num_full_batch_size = len(coords)
        logging.warning(
            f'\t\tCannot fit {num_full_points} points into {self.limit_numpoints} points '
            f'limit. Truncating batch size at {batch_id} out of {num_full_batch_size} with '
            f'{batch_num_points - num_points}.'
        )
        break
      coords_batch.append(coords[batch_id])
      feats_batch.append(feats[batch_id])
      labels_batch.append(point_labels[batch_id])
      bboxes_cls.append(bboxes[batch_id][:, -1].astype(int))
      if self.is_rotation_bbox:
        bboxes_coords.append(bboxes[batch_id][:, :-2])
        bboxes_rotations.append(bboxes[batch_id][:, -2])
      else:
        bboxes_coords.append(bboxes[batch_id][:, :-1])
      last_batch_id = batch_id

    # Concatenate all lists
    bboxes_rotations = bboxes_rotations if self.is_rotation_bbox else None
    coords_sbatch, feats_sbatch, labels_sbatch = \
        ME.utils.sparse_collate(coords_batch, feats_batch, labels_batch)
    datum = {
        'coords': coords_sbatch,
        'input': feats_sbatch.float(),
        'target': labels_sbatch,
        'bboxes_coords': bboxes_coords,
        'bboxes_rotations': bboxes_rotations,
        'bboxes_cls': bboxes_cls,
        'last_batch_id': last_batch_id
    }

    # Precompute detection targets
    if self.config.preload_anchor_data:
      backbone_shapes = self._get_backbone_shapes(datum['coords'])
      anchors, anchor_coords = detection_utils.generate_pyramid_anchors(
          self.config.rpn_anchor_scales, self.anchor_ratios, backbone_shapes,
          self.config.rpn_strides, get_negative_anchors=self.config.load_sparse_gt_data)
      bboxes_normalized = [detection_utils.normalize_boxes(b, self.config.max_ptc_size)
                           for b in bboxes_coords]
      datum.update({
          'backbone_shapes': backbone_shapes,
          'bboxes_normalized': bboxes_normalized,
      })

      if self.config.load_sparse_gt_data:
        anchor_match_coords, sparse_anchor_centers, sparse_anchor_coords, sparse_rpn_match, \
            sparse_rpn_bbox, sparse_rpn_rotation, sparse_rpn_cls = self._get_sparse_rpn_targets(
                coords_batch, anchors, anchor_coords, bboxes_coords, bboxes_rotations, bboxes_cls,
                self.config.rpn_strides)
        datum.update({
            'anchor_match_coords': anchor_match_coords,
            'sparse_anchor_centers': sparse_anchor_centers,
            'sparse_anchor_coords': sparse_anchor_coords,
            'sparse_rpn_match': sparse_rpn_match,
            'sparse_rpn_bbox': sparse_rpn_bbox,
            'sparse_rpn_rotation': sparse_rpn_rotation,
            'sparse_rpn_cls': sparse_rpn_cls,
        })
      else:
        rpn_match, rpn_bbox, rpn_rotation, rpn_cls = self._get_rpn_targets(
            anchors, bboxes_coords, bboxes_rotations, bboxes_cls)
        datum.update({
            'anchors': anchors,
            'rpn_match': rpn_match,
            'rpn_bbox': rpn_bbox,
            'rpn_rotation': rpn_rotation,
            'rpn_cls': rpn_cls,
        })
    return datum


class cflt_collate_fn_factory:
  """Generates collate function for coords, feats, labels, point_clouds, transformations.

    Args:
      limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                       size so that the number of input coordinates is below limit_numpoints.
  """

  def __init__(self, is_rotation_bbox, limit_numpoints, config):
    self.limit_numpoints = limit_numpoints
    self.config = config
    self.cfl_collate_fn = cfl_collate_fn_factory(
        is_rotation_bbox, self.limit_numpoints, self.config)

  def __call__(self, list_data):
    coords, feats, point_labels, bboxes, pointclouds, transformations = list(zip(*list_data))
    datum = self.cfl_collate_fn(list(zip(coords, feats, point_labels, bboxes)))
    num_truncated_batch = datum['last_batch_id'] + 1

    pointclouds_batch = pointclouds[:num_truncated_batch]
    transformations_batch = transformations[:num_truncated_batch]

    pointclouds_batch = [torch.from_numpy(pc) for pc in pointclouds_batch]
    transformations_batch = [torch.from_numpy(t) for t in transformations_batch]

    datum['pointcloud'] = pointclouds_batch
    datum['transformation'] = transformations_batch

    return datum


def elastic_distortion(pointcloud, granularity, magnitude):
  """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
  """
  blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
  blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
  blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
  coords = pointcloud[:, :3]
  coords_min = coords.min(0)

  # Create Gaussian noise tensor of the size given by granularity.
  noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
  noise = np.random.randn(*noise_dim, 3).astype(np.float32)

  # Smoothing.
  for _ in range(2):
    noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
    noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
    noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

  # Trilinear interpolate noise filters for each spatial dimensions.
  ax = [
      np.linspace(d_min, d_max, d)
      for d_min, d_max, d in zip(coords_min - granularity, coords_min
                                 + granularity * (noise_dim - 2), noise_dim)
  ]
  interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
  pointcloud[:, :3] = coords + interp(coords) * magnitude
  return pointcloud
