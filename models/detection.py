import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

import lib.detection_utils as utils
from models.model import Model


class PyramidROIAlign(Model):
  def __init__(self, in_channels, in_pixel_dists, pool_size, config, D=3, **kwargs):
    super().__init__(in_channels, in_channels, config, D, **kwargs)
    self.pool_size = pool_size
    self.in_pixel_dists = in_pixel_dists
    self.num_pyramids = len(in_pixel_dists)
    self.OUT_PIXEL_DIST = in_pixel_dists[0]

  def _gen_cache(self, x):
    return None

  def _get_level_feat(self, x, cache, pyramid_level, batch_idx):
    level_feat = x[pyramid_level][batch_idx]
    level_min = torch.zeros(3).to(level_feat)
    return level_feat, level_min

  def forward(self, batch_rois, batch_rotations, x):
    batch_rois_aligned = []
    cache = self._gen_cache(x)
    for batch_idx, rois in enumerate(batch_rois):
      rois_scale = np.cbrt(np.prod(rois[:, 3:] - rois[:, :3], 1))
      rois_level = np.floor(
          self.config.fpn_base_level + np.log2(rois_scale / self.config.fpn_max_scale))
      rois_level = np.clip(rois_level, 0, self.num_pyramids - 1).astype(int)
      pyramid_idxs_levels = []
      pyramid_feats_levels = []
      for pyramid_level in range(self.num_pyramids):
        pyramid_idxs = np.where(rois_level == pyramid_level)[0]
        batch_size = pyramid_idxs.size
        if batch_size == 0:
          continue
        pyramid_idxs_levels.append(pyramid_idxs)
        level_feat, level_min = self._get_level_feat(x, cache, pyramid_level, batch_idx)
        level_shape = torch.tensor(level_feat.shape[1:]).to(level_feat)
        if self.config.roialign_align_corners:
          level_shape -= 1
        level_feat = level_feat.permute(0, 3, 2, 1)
        level_rois = torch.from_numpy(rois[pyramid_idxs]).to(level_feat)
        grid_size = (level_rois[:, 3:] - level_rois[:, :3]) / self.pool_size
        grid_base = level_rois[:, :3, None].repeat(1, 1, self.pool_size)
        grid_step = torch.arange(0, self.pool_size).to(grid_size) + 0.5
        grid_space = grid_base + (grid_size[:, :, None].repeat(1, 1, self.pool_size) * grid_step)
        grid_x = grid_space[:, 0].reshape(batch_size, self.pool_size, 1, 1).repeat(
            1, 1, self.pool_size, self.pool_size).unsqueeze(-1)
        grid_y = grid_space[:, 1].reshape(batch_size, 1, self.pool_size, 1).repeat(
            1, self.pool_size, 1, self.pool_size).unsqueeze(-1)
        grid_z = grid_space[:, 2].reshape(batch_size, 1, 1, self.pool_size).repeat(
            1, self.pool_size, self.pool_size, 1).unsqueeze(-1)
        level_grids = torch.cat((grid_x, grid_y, grid_z), -1)
        if batch_rotations is not None:
          grid_centers = (level_rois[:, 3:] + level_rois[:, :3]).reshape(batch_size, 1, 1, 1, 3) / 2
          level_grids -= grid_centers
          level_grids = level_grids.reshape(batch_size, -1, 3).permute(0, 2, 1)
          rot = batch_rotations[batch_idx][pyramid_idxs]
          level_grids = utils.apply_rotations(level_grids, rot).permute(0, 2, 1).reshape(
              batch_size, self.pool_size, self.pool_size, self.pool_size, 3) + grid_centers
        level_grids -= level_min
        level_grids /= self.in_pixel_dists[pyramid_level]
        level_grids = level_grids / torch.clamp(level_shape, min=1) * 2 - 1
        pyramid_feats_levels.append(
            F.grid_sample(
                level_feat.unsqueeze(0),
                level_grids.reshape(1, -1, self.pool_size, self.pool_size, 3),
                align_corners=self.config.roialign_align_corners,
                padding_mode='zeros').reshape(level_feat.shape[0], -1, self.pool_size,
                                              self.pool_size, self.pool_size).transpose(0, 1))
      if pyramid_feats_levels:
        pyramid_feats = torch.cat(pyramid_feats_levels)
        pyramid_idxs = torch.from_numpy(
            np.concatenate(pyramid_idxs_levels).argsort()).to(pyramid_feats).long()
        rois_aligned = torch.index_select(pyramid_feats, 0, pyramid_idxs)
        batch_rois_aligned.append(rois_aligned)
    return torch.cat(batch_rois_aligned)


class FeaturePyramidClassifierNetwork(Model):

  def __init__(self, in_channels, in_pixel_dists, num_class, config, is_rotation_bbox,
               rotation_criterion, D=3, **kwargs):
    super().__init__(in_channels, num_class + 1, config, D, **kwargs)
    self.is_rotation_bbox = is_rotation_bbox
    self.rotation_criterion = rotation_criterion
    self.pool_size = config.refinement_roialign_poolsize
    self.in_pixel_dists = in_pixel_dists
    self.num_class = num_class
    self.network_initialization(in_channels, config, D)
    self.weight_initialization()

  def network_initialization(self, in_channels, config, D):
    self.roialign = PyramidROIAlign(in_channels, self.in_pixel_dists, self.pool_size, config)
    self.fc1 = nn.Linear(self.in_channels * self.pool_size ** D,
                         config.ref_classification_feat_size)
    self.bn1 = nn.BatchNorm1d(config.ref_classification_feat_size)
    self.fc2 = nn.Linear(config.ref_classification_feat_size, config.ref_classification_feat_size)
    self.bn2 = nn.BatchNorm1d(config.ref_classification_feat_size)
    self.classifier = nn.Linear(config.ref_classification_feat_size, self.num_class + 1)
    self.refinement = nn.Linear(config.ref_classification_feat_size, self.num_class * D * 2)
    if self.is_rotation_bbox:
      self.rotation = nn.Linear(
          config.ref_classification_feat_size, self.num_class * self.rotation_criterion.NUM_OUTPUT)

  def forward(self, rois, rotations, fpn_outputs):
    batch_size = sum(roi.shape[0] for roi in rois)
    if batch_size == 0:
      return None, None, None, None
    x = self.roialign(rois, rotations, fpn_outputs)
    if torch.any(torch.isnan(x)):
      return None, None, None, None
    x = x.reshape(batch_size, self.in_channels * self.pool_size ** self.D)
    x = F.relu(self.bn1(self.fc1(x)))
    x = F.relu(self.bn2(self.fc2(x)))
    ref_class_logits = self.classifier(x)
    ref_class_probs = F.softmax(ref_class_logits, dim=1)
    ref_bbox = self.refinement(x).reshape(batch_size, self.num_class, self.D * 2)
    if self.is_rotation_bbox:
      ref_rot = self.rotation(x).reshape(batch_size, self.num_class,
                                         self.rotation_criterion.NUM_OUTPUT)
    else:
      ref_rot = None
    return ref_class_logits, ref_class_probs, ref_bbox, ref_rot


class FeatureUpsampleNetwork(Model):
  """A network which upsamples and builds a feature pyramid of different strides."""
  NUM_PYRAMIDS = 4

  def __init__(self, in_channels, in_pixel_dists, config, D=3, **kwargs):
    assert self.NUM_PYRAMIDS > 0 and config.upsample_feat_size > 0
    assert len(in_channels) == len(in_pixel_dists) == self.NUM_PYRAMIDS
    super().__init__(in_channels, config.upsample_feat_size, config, D, **kwargs)
    self.in_pixel_dists = in_pixel_dists
    self.OUT_PIXEL_DIST = self.in_pixel_dists
    self.network_initialization(in_channels, config, D)
    self.weight_initialization()

  def network_initialization(self, in_channels, config, D):
    self.conv_feat1 = nn.Conv3d(in_channels[0], config.upsample_feat_size, 3, padding=1)
    self.conv_feat2 = nn.Conv3d(in_channels[1], config.upsample_feat_size, 3, padding=1)
    self.conv_feat3 = nn.Conv3d(in_channels[2], config.upsample_feat_size, 3, padding=1)
    self.conv_feat4 = nn.Conv3d(in_channels[3], config.upsample_feat_size, 3, padding=1)
    self.bn_feat1 = nn.BatchNorm3d(config.upsample_feat_size)
    self.bn_feat2 = nn.BatchNorm3d(config.upsample_feat_size)
    self.bn_feat3 = nn.BatchNorm3d(config.upsample_feat_size)
    self.bn_feat4 = nn.BatchNorm3d(config.upsample_feat_size)
    self.conv_up2 = nn.ConvTranspose3d(config.upsample_feat_size, config.upsample_feat_size, 2, 2)
    self.conv_up3 = nn.ConvTranspose3d(config.upsample_feat_size, config.upsample_feat_size, 2, 2)
    self.conv_up4 = nn.ConvTranspose3d(config.upsample_feat_size, config.upsample_feat_size, 2, 2)
    self.bn_up2 = nn.BatchNorm3d(config.upsample_feat_size)
    self.bn_up3 = nn.BatchNorm3d(config.upsample_feat_size)
    self.bn_up4 = nn.BatchNorm3d(config.upsample_feat_size)
    self.conv_final1 = nn.Conv3d(
        config.upsample_feat_size, config.upsample_feat_size, 3, stride=1, padding=1)
    self.conv_final2 = nn.Conv3d(
        config.upsample_feat_size, config.upsample_feat_size, 3, stride=1, padding=1)
    self.conv_final3 = nn.Conv3d(
        config.upsample_feat_size, config.upsample_feat_size, 3, stride=1, padding=1)
    self.conv_final4 = nn.Conv3d(
        config.upsample_feat_size, config.upsample_feat_size, 3, stride=1, padding=1)
    self.bn_final1 = nn.BatchNorm3d(config.upsample_feat_size)
    self.bn_final2 = nn.BatchNorm3d(config.upsample_feat_size)
    self.bn_final3 = nn.BatchNorm3d(config.upsample_feat_size)
    self.bn_final4 = nn.BatchNorm3d(config.upsample_feat_size)

  def forward(self, backbone_outputs):
    # Precompute input tensor shapes
    output_coords = torch.cat(backbone_outputs[-1].decomposed_coordinates, 0)
    voxel_shape = output_coords.max(0)[0] + self.in_pixel_dists[-1]
    # Enumerate network over pyramids.
    fpn_outputs = []
    pyramid_output = None
    for layer_idx in reversed(range(len(backbone_outputs))):
      dense_tensor = utils.sparse2dense(
          backbone_outputs[layer_idx], self.in_pixel_dists[layer_idx], voxel_shape)
      conv_feat = self.get_layer('conv_feat', layer_idx)
      bn_feat = self.get_layer('bn_feat', layer_idx)
      fpn_feat = F.relu(bn_feat(conv_feat(dense_tensor)))
      if pyramid_output is not None:
        fpn_feat += pyramid_output
      conv_up = self.get_layer('conv_up', layer_idx)
      if conv_up is not None:
        bn_up = self.get_layer('bn_up', layer_idx)
        pyramid_output = F.relu(bn_up(conv_up(fpn_feat)))
      conv_final = self.get_layer('conv_final', layer_idx)
      bn_final = self.get_layer('bn_final', layer_idx)
      fpn_outputs.insert(0, F.relu(bn_final(conv_final(fpn_feat))))
    return fpn_outputs


class RegionProposalNetwork(Model):
  """A network which takes a set of FPN outputs and anchors to generate region proposals."""
  OUT_PIXEL_DIST = 1

  def __init__(self, in_channels, anchors_per_location, config, is_rotation_bbox,
               rotation_criterion, num_class, D=3, **kwargs):
    assert config.proposal_feat_size > 1
    super().__init__(in_channels, anchors_per_location, config, D, **kwargs)
    self.is_rotation_bbox = is_rotation_bbox
    self.rotation_criterion = rotation_criterion
    self.network_initialization(in_channels, config, D)
    self.weight_initialization()

  def network_initialization(self, in_channels, config, D):
    self.conv1 = nn.Conv3d(in_channels, config.proposal_feat_size, 3, padding=1)
    self.bn1 = nn.BatchNorm3d(config.proposal_feat_size)
    self.final_class_logits = nn.Conv3d(config.proposal_feat_size, self.out_channels * 2, 1)
    self.final_bbox = nn.Conv3d(config.proposal_feat_size, self.out_channels * 6, 1)
    if self.is_rotation_bbox:
      self.final_rotation = nn.Conv3d(
          config.proposal_feat_size, self.out_channels * self.rotation_criterion.NUM_OUTPUT, 1)

  def batch_non_maximum_suppression(self, batch_boxes, batch_rotations, batch_scores,
                                    proposal_count):
    batch_nms = []
    batch_nms_scores = []
    batch_rots = None if batch_rotations is None else []
    for i, (boxes, scores) in enumerate(zip(batch_boxes, batch_scores)):
      rotation = None
      if batch_rots is not None:
        rotation = batch_rotations[i]
      nms_roi, nms_rot, nms_score = utils.non_maximum_suppression(
          boxes, rotation, scores, self.config.rpn_nms_threshold, proposal_count,
          self.config.rpn_rot_nms, self.config.rpn_aggregate_overlap)
      padded_boxes = torch.zeros((proposal_count - len(nms_score), boxes.shape[1])).to(boxes)
      batch_nms.append(torch.cat((nms_roi, padded_boxes), 0))
      batch_nms_scores.append(nms_score)
      if batch_rots is not None:
        batch_rots.append(nms_rot)
    return torch.stack(batch_nms), batch_rots, batch_nms_scores

  def get_proposal(self, rpn_probs, deltas, rotation, anchors, num_proposals):
    assert deltas.shape[1:] == anchors.shape
    scores = rpn_probs[:, :, 1]
    rpn_bbox_std = np.reshape(self.config.rpn_bbox_std, (1, 1, len(self.config.rpn_bbox_std)))
    deltas *= torch.from_numpy(rpn_bbox_std).to(deltas)
    anchors = torch.from_numpy(np.broadcast_to(anchors, deltas.shape)).to(deltas)
    pre_nms_limit = min(self.config.rpn_pre_nms_limit, anchors.shape[1])
    scores, ix = torch.topk(scores, pre_nms_limit, sorted=True)
    ix = [i[s > self.config.rpn_pre_nms_min_confidence] for s, i in zip(scores, ix)]
    scores = [s[s > self.config.rpn_pre_nms_min_confidence] for s, i in zip(scores, ix)]
    deltas = [torch.index_select(o, 0, i) for o, i in zip(deltas, ix)]
    anchors = [torch.index_select(o, 0, i) for o, i in zip(anchors, ix)]
    boxes = [utils.apply_box_deltas(a, d, self.config.normalize_bbox)
             for a, d in zip(anchors, deltas)]
    if rotation is not None:
      with torch.no_grad():
        rotation = [self.rotation_criterion.pred(torch.index_select(o, 0, i))
                    for o, i in zip(rotation, ix)]
    rpn_proposal, rotation, rpn_scores = self.batch_non_maximum_suppression(
        boxes, rotation, scores, num_proposals)
    return rpn_proposal, rotation, rpn_scores

  @staticmethod
  def _reshape_proposal_feat(x, dim):
    return x.permute(0, *range(2, len(x.shape)), 1).reshape(x.shape[0], -1, dim)

  def _forward_stride(self, x):
    feat = F.relu(self.bn1(self.conv1(x)))
    rpn_class_logits = self._reshape_proposal_feat(self.final_class_logits(feat), 2)
    rpn_probs = F.softmax(rpn_class_logits, dim=2)
    rpn_bbox = self._reshape_proposal_feat(self.final_bbox(feat), 6)
    if self.is_rotation_bbox:
      rpn_rotation = self._reshape_proposal_feat(self.final_rotation(feat),
                                                 self.rotation_criterion.NUM_OUTPUT)
      return rpn_class_logits, rpn_probs, rpn_bbox, rpn_rotation
    return rpn_class_logits, rpn_probs, rpn_bbox

  def forward(self, fpn_outputs, anchors, num_proposals, generate_proposal=True):
    # Forward network for each layer of the pyramid.
    rpn_outputs = [torch.cat(o, 1) for o in zip(*[self._forward_stride(p) for p in fpn_outputs])]
    if self.is_rotation_bbox:
      rpn_class_logits, rpn_probs, rpn_bbox, rpn_rotation = rpn_outputs
    else:
      rpn_class_logits, rpn_probs, rpn_bbox = rpn_outputs
      rpn_rotation = None
    if generate_proposal:
      # Get bounding boxes from detection output.
      anchors = utils.normalize_boxes(anchors, self.config.max_ptc_size)
      rpn_rois, rpn_rois_rotation, rpn_scores = self.get_proposal(
          rpn_probs, rpn_bbox, rpn_rotation, anchors, num_proposals)
    else:
      rpn_rois, rpn_rois_rotation, rpn_scores = None, None, None
    return rpn_class_logits, rpn_bbox, rpn_rotation, rpn_rois, rpn_rois_rotation, rpn_scores


class FakeSparseGenerativeFeatureUpsampleNetwork(Model):

  def __init__(self, in_channels, in_pixel_dists, config, D=3, **kwargs):
    assert config.upsample_feat_size > 0
    assert len(in_channels) == len(in_pixel_dists)
    super().__init__(in_channels, config.upsample_feat_size, config, D, **kwargs)
    self.in_pixel_dists = in_pixel_dists
    self.OUT_PIXEL_DIST = self.in_pixel_dists
    self.network_initialization(in_channels, config, D)
    self.weight_initialization()

  def network_initialization(self, in_channels, config, D):
    self.conv_feat1 = nn.Sequential(
        ME.MinkowskiConvolution(
            in_channels[0], config.upsample_feat_size, kernel_size=1, dimension=3),
        ME.MinkowskiBatchNorm(config.upsample_feat_size),
        ME.MinkowskiELU())

    self.conv_feat2 = nn.Sequential(
        ME.MinkowskiConvolution(
            in_channels[1], config.upsample_feat_size, kernel_size=1, dimension=3),
        ME.MinkowskiBatchNorm(config.upsample_feat_size),
        ME.MinkowskiELU())

    self.conv_feat3 = nn.Sequential(
        ME.MinkowskiConvolution(
            in_channels[2], config.upsample_feat_size, kernel_size=1, dimension=3),
        ME.MinkowskiBatchNorm(config.upsample_feat_size),
        ME.MinkowskiELU())

    self.conv_feat4 = nn.Sequential(
        ME.MinkowskiConvolution(
            in_channels[3], config.upsample_feat_size, kernel_size=1, dimension=3),
        ME.MinkowskiBatchNorm(config.upsample_feat_size),
        ME.MinkowskiELU())

  def forward(self, backbone_outputs, match_coords, is_train):
    # Enumerate network over pyramids.
    fpn_outputs = []
    num_layers = len(backbone_outputs)

    for layer_idx in reversed(range(num_layers)):
      conv_feat_layer = self.get_layer('conv_feat', layer_idx)
      fpn_output = conv_feat_layer(backbone_outputs[layer_idx])
      fpn_outputs.insert(0, fpn_output)

    return fpn_outputs, None, None


class SparseGenerativeNoPruningFeatureUpsampleNetwork(Model):

  def __init__(self, in_channels, in_pixel_dists, config, D=3, **kwargs):
    assert config.upsample_feat_size > 0
    assert len(in_channels) == len(in_pixel_dists)
    super().__init__(in_channels, config.upsample_feat_size, config, D, **kwargs)
    self.in_pixel_dists = in_pixel_dists
    self.OUT_PIXEL_DIST = self.in_pixel_dists
    self.network_initialization(in_channels, config, D)
    self.weight_initialization()

  def network_initialization(self, in_channels, config, D):
    up_kernel_size = 3
    self.conv_up1 = nn.Sequential(
        ME.MinkowskiConvolutionTranspose(
            in_channels[0], in_channels[0], kernel_size=up_kernel_size, stride=2,
            generate_new_coords=True, dimension=3),
        ME.MinkowskiBatchNorm(in_channels[0]),
        ME.MinkowskiELU())

    self.conv_up2 = nn.Sequential(
        ME.MinkowskiConvolutionTranspose(
            in_channels[1], in_channels[0], kernel_size=up_kernel_size, stride=2,
            generate_new_coords=True, dimension=3),
        ME.MinkowskiBatchNorm(in_channels[0]),
        ME.MinkowskiELU())

    self.conv_up3 = nn.Sequential(
        ME.MinkowskiConvolutionTranspose(
            in_channels[2], in_channels[1], kernel_size=up_kernel_size, stride=2,
            generate_new_coords=True, dimension=3),
        ME.MinkowskiBatchNorm(in_channels[1]),
        ME.MinkowskiELU())

    self.conv_up4 = nn.Sequential(
        ME.MinkowskiConvolutionTranspose(
            in_channels[3], in_channels[2], kernel_size=up_kernel_size, stride=2,
            generate_new_coords=True, dimension=3),
        ME.MinkowskiBatchNorm(in_channels[2]),
        ME.MinkowskiELU())

    self.conv_feat1 = nn.Sequential(
        ME.MinkowskiConvolution(
            in_channels[0], config.upsample_feat_size, kernel_size=1, dimension=3),
        ME.MinkowskiBatchNorm(config.upsample_feat_size),
        ME.MinkowskiELU())

    self.conv_feat2 = nn.Sequential(
        ME.MinkowskiConvolution(
            in_channels[1], config.upsample_feat_size, kernel_size=1, dimension=3),
        ME.MinkowskiBatchNorm(config.upsample_feat_size),
        ME.MinkowskiELU())

    self.conv_feat3 = nn.Sequential(
        ME.MinkowskiConvolution(
            in_channels[2], config.upsample_feat_size, kernel_size=1, dimension=3),
        ME.MinkowskiBatchNorm(config.upsample_feat_size),
        ME.MinkowskiELU())

    self.conv_feat4 = nn.Sequential(
        ME.MinkowskiConvolution(
            in_channels[3], config.upsample_feat_size, kernel_size=1, dimension=3),
        ME.MinkowskiBatchNorm(config.upsample_feat_size),
        ME.MinkowskiELU())

  def forward(self, backbone_outputs, match_coords, is_train):
    # Enumerate network over pyramids.
    fpn_outputs = []
    pyramid_output = None
    num_layers = len(backbone_outputs)

    for layer_idx in reversed(range(num_layers)):
      conv_feat_layer = self.get_layer('conv_feat', layer_idx)
      conv_up_layer = self.get_layer('conv_up', layer_idx)

      # Current feature
      curr_feat = backbone_outputs[layer_idx]

      # Add previous layer output
      if pyramid_output is not None:
        assert pyramid_output.tensor_stride == curr_feat.tensor_stride
        curr_feat = curr_feat + pyramid_output

      # Upsample
      pyramid_output = conv_up_layer(curr_feat)

      # Generate final feature for current level
      fpn_output = conv_feat_layer(curr_feat)

      # Post processing
      fpn_outputs.insert(0, fpn_output)

    return fpn_outputs, None, None


class SparseGenerativeFeatureUpsampleNetwork(Model):

  def __init__(self, in_channels, in_pixel_dists, config, D=3, **kwargs):
    assert config.upsample_feat_size > 0
    assert len(in_channels) == len(in_pixel_dists)
    super().__init__(in_channels, config.upsample_feat_size, config, D, **kwargs)
    self.in_pixel_dists = in_pixel_dists
    self.OUT_PIXEL_DIST = self.in_pixel_dists
    self.network_initialization(in_channels, config, D)
    self.weight_initialization()

  def network_initialization(self, in_channels, config, D):
    up_kernel_size = 3
    self.conv_up1 = nn.Sequential(
        ME.MinkowskiConvolutionTranspose(
            in_channels[0], in_channels[0], kernel_size=up_kernel_size, stride=2,
            generate_new_coords=True, dimension=3),
        ME.MinkowskiBatchNorm(in_channels[0]),
        ME.MinkowskiELU())

    self.conv_up2 = nn.Sequential(
        ME.MinkowskiConvolutionTranspose(
            in_channels[1], in_channels[0], kernel_size=up_kernel_size, stride=2,
            generate_new_coords=True, dimension=3),
        ME.MinkowskiBatchNorm(in_channels[0]),
        ME.MinkowskiELU())

    self.conv_up3 = nn.Sequential(
        ME.MinkowskiConvolutionTranspose(
            in_channels[2], in_channels[1], kernel_size=up_kernel_size, stride=2,
            generate_new_coords=True, dimension=3),
        ME.MinkowskiBatchNorm(in_channels[1]),
        ME.MinkowskiELU())

    self.conv_up4 = nn.Sequential(
        ME.MinkowskiConvolutionTranspose(
            in_channels[3], in_channels[2], kernel_size=up_kernel_size, stride=2,
            generate_new_coords=True, dimension=3),
        ME.MinkowskiBatchNorm(in_channels[2]),
        ME.MinkowskiELU())

    self.conv_feat1 = nn.Sequential(
        ME.MinkowskiConvolution(
            in_channels[0], config.upsample_feat_size, kernel_size=1, dimension=3),
        ME.MinkowskiBatchNorm(config.upsample_feat_size),
        ME.MinkowskiELU())

    self.conv_feat2 = nn.Sequential(
        ME.MinkowskiConvolution(
            in_channels[1], config.upsample_feat_size, kernel_size=1, dimension=3),
        ME.MinkowskiBatchNorm(config.upsample_feat_size),
        ME.MinkowskiELU())

    self.conv_feat3 = nn.Sequential(
        ME.MinkowskiConvolution(
            in_channels[2], config.upsample_feat_size, kernel_size=1, dimension=3),
        ME.MinkowskiBatchNorm(config.upsample_feat_size),
        ME.MinkowskiELU())

    self.conv_feat4 = nn.Sequential(
        ME.MinkowskiConvolution(
            in_channels[3], config.upsample_feat_size, kernel_size=1, dimension=3),
        ME.MinkowskiBatchNorm(config.upsample_feat_size),
        ME.MinkowskiELU())

    self.conv_cls1 = ME.MinkowskiConvolution(
        config.upsample_feat_size, 2, kernel_size=1, has_bias=True, dimension=3)
    self.conv_cls2 = ME.MinkowskiConvolution(
        config.upsample_feat_size, 2, kernel_size=1, has_bias=True, dimension=3)
    self.conv_cls3 = ME.MinkowskiConvolution(
        config.upsample_feat_size, 2, kernel_size=1, has_bias=True, dimension=3)
    self.conv_cls4 = ME.MinkowskiConvolution(
        config.upsample_feat_size, 2, kernel_size=1, has_bias=True, dimension=3)

    self.elu = ME.MinkowskiELU()
    self.pruning = ME.MinkowskiPruning()

  def forward(self, backbone_outputs, match_coords, is_train):
    # Enumerate network over pyramids.
    fpn_outputs = []
    targets = []
    classifications = []
    pyramid_output = None
    num_layers = len(backbone_outputs)
    if is_train:
      target_coords = [ME.utils.batched_coordinates([match[i][0] for match in match_coords])
                       for i in range(num_layers)]
      ambiguous_coords = [ME.utils.batched_coordinates([match[i][1] for match in match_coords])
                          for i in range(num_layers)]

    for layer_idx in reversed(range(num_layers)):
      conv_feat_layer = self.get_layer('conv_feat', layer_idx)
      conv_cls_layer = self.get_layer('conv_cls', layer_idx)
      conv_up_layer = self.get_layer('conv_up', layer_idx)

      # Current feature
      curr_feat = backbone_outputs[layer_idx]

      # Add previous layer output
      if pyramid_output is not None:
        assert pyramid_output.tensor_stride == curr_feat.tensor_stride
        curr_feat = curr_feat + pyramid_output

      # Two branches: upsample and fpn feature and classification
      # 1. FPN feature & classification
      fpn_feat = conv_feat_layer(curr_feat)
      feat_cls = conv_cls_layer(fpn_feat)
      pred_prob = F.softmax(feat_cls.F, 1)[:, 1]

      # target calculation
      target = None
      if is_train:
        target = torch.zeros(len(fpn_feat), dtype=torch.long)
        pos_ins = utils.map_coordinates(fpn_feat, torch.cat(ambiguous_coords[:layer_idx + 1]),
                                        force_stride=True)[0]
        target[pos_ins] = self.config.ignore_label
        pos_ins = utils.map_coordinates(fpn_feat, torch.cat(target_coords[:layer_idx + 1]),
                                        force_stride=True)[0]
        target[pos_ins] = 1

      # Get keep labels
      keep = (pred_prob > self.config.sfpn_min_confidence).cpu()
      if is_train:  # Force put GT labels within keep
        keep |= target == 1

      if torch.any(keep):
        # Prune and upsample
        pyramid_output = conv_up_layer(self.pruning(curr_feat, keep))
        # Generate final feature for current level
        final_pruned = self.pruning(fpn_feat, keep)
      else:
        pyramid_output = None
        final_pruned = None

      # Post processing
      classifications.insert(0, feat_cls)
      targets.insert(0, target)
      fpn_outputs.insert(0, final_pruned)

    return fpn_outputs, targets, classifications


class SparseRegionProposalNetwork(RegionProposalNetwork):

  def network_initialization(self, in_channels, config, D):
    self.conv1 = ME.MinkowskiConvolution(
        in_channels, config.proposal_feat_size, kernel_size=1, dimension=3)
    self.bn1 = ME.MinkowskiInstanceNorm(config.proposal_feat_size)
    self.conv2 = ME.MinkowskiConvolution(
        config.proposal_feat_size, config.proposal_feat_size, kernel_size=1, dimension=3)
    self.bn2 = ME.MinkowskiInstanceNorm(config.proposal_feat_size)
    self.final_class_logits = ME.MinkowskiConvolution(
        config.proposal_feat_size, self.out_channels * 2, kernel_size=1, dimension=3, has_bias=True)
    self.final_bbox = ME.MinkowskiConvolution(
        config.proposal_feat_size, self.out_channels * 6, kernel_size=1, dimension=3, has_bias=True)
    self.elu = ME.MinkowskiELU()
    self.softmax = ME.MinkowskiSoftmax()
    if self.is_rotation_bbox:
      self.final_rotation = ME.MinkowskiConvolution(
          config.proposal_feat_size, self.out_channels * self.rotation_criterion.NUM_OUTPUT,
          kernel_size=1, dimension=3, has_bias=True)

  def get_proposal(self, lrpn_probs, ldeltas, lrotation, lanchors, num_proposals):
    with torch.no_grad():
      assert len(lrpn_probs) == len(ldeltas) == len(lrotation) == len(lanchors)
      rpn_scores = []
      rpn_boxes = []
      rpn_rotations = []
      rpn_batch_idxs = []
      rpn2anchor_maps = []
      for rpn_probs, deltas, rotation, (anchor_coords, anchor_feats) in zip(
              lrpn_probs, ldeltas, lrotation, lanchors):
        if rpn_probs is None:
          rpn2anchor_maps.append((None, None))
          continue
        assert rpn_probs.coords_key == deltas.coords_key
        assert rpn_probs.F.shape[1] / 2 == deltas.F.shape[1] / 6
        assert deltas.F.shape[-1] == anchor_feats.shape[-1]
        rpn2anchor, anchor2rpn = utils.map_coordinates(deltas, anchor_coords, check_input_map=True)
        rpn2anchor_maps.append((rpn2anchor, anchor2rpn))
        rpn_batch_idxs.append(deltas.coords[rpn2anchor][:, 0])
        rpn_scores.append(rpn_probs.F[rpn2anchor].reshape(-1, 2)[:, 1])
        rpn_bbox_std = torch.from_numpy(np.expand_dims(self.config.rpn_bbox_std, 0)).to(deltas.F)
        deltas = deltas.F[rpn2anchor].reshape(-1, 6) * rpn_bbox_std
        anchors = anchor_feats[anchor2rpn].reshape(-1, 6)
        boxes = utils.apply_box_deltas(anchors.to(deltas), deltas, self.config.normalize_bbox)
        rpn_boxes.append(boxes)
        if rotation is not None:
          num_rot_output = self.rotation_criterion.NUM_OUTPUT
          assert rpn_probs.coords_key == rotation.coords_key
          assert rpn_probs.F.shape[1] / 2 == rotation.F.shape[1] / num_rot_output
          rpn_rotations.append(
              self.rotation_criterion.pred(rotation.F[rpn2anchor].reshape(-1, num_rot_output)))
      if not rpn_scores:
        return None, None, None, rpn2anchor_maps
      all_scores = torch.cat(rpn_scores)
      all_boxes = torch.cat(rpn_boxes)
      all_batch_idxs = torch.cat(rpn_batch_idxs).repeat_interleave(7)
      rotations = None
      if rpn_rotations:
        all_rotations = torch.cat(rpn_rotations)
        rotations = []
      boxes = []
      scores = []
      for i in range(all_batch_idxs.max().item() + 1):
        batch_mask = all_batch_idxs == i
        batch_scores = all_scores[batch_mask]
        confidence_mask = batch_scores > self.config.rpn_pre_nms_min_confidence
        pre_nms_limit = min(self.config.rpn_pre_nms_limit, confidence_mask.sum())
        batch_scores, ix = torch.topk(batch_scores[confidence_mask], pre_nms_limit, sorted=True)
        scores.append(batch_scores)
        boxes.append(all_boxes[batch_mask][confidence_mask][ix])
        if rotations is not None:
          rotations.append(all_rotations[batch_mask][confidence_mask][ix])
      rpn_proposal, rotation, rpn_scores = self.batch_non_maximum_suppression(
          boxes, rotations, scores, num_proposals)
      return rpn_proposal, rotation, rpn_scores, rpn2anchor_maps

  def _forward_stride(self, x):
    rpn_class_logits, rpn_probs, rpn_bbox, rpn_rotation = None, None, None, None
    if x is not None:
      x = self.elu(self.bn1(self.conv1(x)))
      feat = self.elu(self.bn2(self.conv2(x)))
      rpn_class_logits = self.final_class_logits(feat)
      rpn_class_logits_shape = rpn_class_logits.F.shape
      rpn_probs = ME.SparseTensor(
          F.softmax(rpn_class_logits.F.reshape(-1, 2), dim=1).reshape(*rpn_class_logits_shape),
          coords_key=rpn_class_logits.coords_key, coords_manager=rpn_class_logits.coords_man)
      rpn_bbox = self.final_bbox(feat)
      if self.is_rotation_bbox:
        rpn_rotation = self.final_rotation(feat)
    return rpn_class_logits, rpn_probs, rpn_bbox, rpn_rotation

  def forward(self, fpn_outputs, anchor_coord, anchor_feat, num_proposals, generate_proposal=True):
    # Forward network for each layer of the pyramid.
    rpn_class_logits, rpn_probs, rpn_bbox, rpn_rotation = zip(
        *[self._forward_stride(p) for p in fpn_outputs])
    if generate_proposal:
      # Get bounding boxes from detection output.
      anchors = [(c, torch.from_numpy(utils.normalize_boxes(f.numpy(), self.config.max_ptc_size)))
                 for c, f in zip(anchor_coord, anchor_feat)]
      rpn_rois, rpn_rois_rotation, rpn_scores, rpn2anchor_maps = self.get_proposal(
          rpn_probs, rpn_bbox, rpn_rotation, anchors, num_proposals)
    else:
      rpn_rois, rpn_rois_rotation, rpn_scores, rpn2anchor_maps = None, None, None, None
    return (rpn_class_logits, rpn_bbox, rpn_rotation, rpn_rois, rpn_rois_rotation, rpn_scores,
            rpn2anchor_maps)


class SparseRegionProposalClassifierNetwork(Model):
  """A network which takes a set of FPN outputs and anchors to generate region proposals."""
  OUT_PIXEL_DIST = 1

  def __init__(self, in_channels, anchors_per_location, config, is_rotation_bbox,
               rotation_criterion, num_class, D=3, **kwargs):
    assert config.proposal_feat_size > 1
    super().__init__(in_channels, anchors_per_location, config, D, **kwargs)
    self.num_class = num_class
    self.is_rotation_bbox = is_rotation_bbox
    self.rotation_criterion = rotation_criterion
    self.anchor_sizes = [scale / np.sqrt(np.array(self.config.rpn_anchor_ratios)) / 2
                         for scale in self.config.rpn_anchor_scales]
    self.network_initialization(in_channels, config, D)
    self.weight_initialization()

  def network_initialization(self, in_channels, config, D):
    self.conv1 = ME.MinkowskiConvolution(
        in_channels, config.proposal_feat_size, kernel_size=1, dimension=3)
    self.bn1 = ME.MinkowskiInstanceNorm(config.proposal_feat_size)
    self.conv2 = ME.MinkowskiConvolution(
        config.proposal_feat_size, config.proposal_feat_size, kernel_size=1, dimension=3)
    self.bn2 = ME.MinkowskiInstanceNorm(config.proposal_feat_size)
    self.final_class_logits = ME.MinkowskiConvolution(
        config.proposal_feat_size, self.out_channels * 2, kernel_size=1, dimension=3,
        has_bias=True)
    self.final_semantic_logits = ME.MinkowskiConvolution(
        config.proposal_feat_size, self.out_channels * self.num_class, kernel_size=1, dimension=3,
        has_bias=True)
    self.final_bbox = ME.MinkowskiConvolution(
        config.proposal_feat_size, self.out_channels * 6, kernel_size=1, dimension=3, has_bias=True)
    self.elu = ME.MinkowskiELU()
    self.softmax = ME.MinkowskiSoftmax()
    if self.is_rotation_bbox:
      self.final_rotation = ME.MinkowskiConvolution(
          config.proposal_feat_size, self.out_channels * self.rotation_criterion.NUM_OUTPUT,
          kernel_size=1, dimension=3, has_bias=True)

  def batch_non_maximum_suppression(self, batch_boxes, batch_rotations, batch_classes, batch_scores,
                                    batch_return_scores, proposal_count):
    batch_nms = []
    batch_nms_scores = []
    batch_rots = None if batch_rotations is None else []
    for i, (boxes, classes, scores, return_scores) in enumerate(zip(
            batch_boxes, batch_classes, batch_scores, batch_return_scores)):
      rotation = None
      if batch_rotations is not None:
        rotation = batch_rotations[i]
      keep = torch.where(classes >= 0)[0].cpu().numpy()
      if self.config.detection_min_confidence:
        conf_keep = torch.where(scores > self.config.detection_min_confidence)[0]
        keep = np.array(list(set(conf_keep.cpu().numpy()).intersection(keep)))
      if keep.size == 0:
        batch_nms.append(torch.zeros((0, 7)).to(boxes))
        batch_nms_scores.append(torch.zeros(0).to(scores))
        if batch_rots is not None:
          batch_rots.append(torch.zeros(0).to(rotation))
      else:
        pre_nms_class_ids = classes[keep]
        pre_nms_scores = scores[keep]
        pre_nms_return_scores = return_scores[keep]
        pre_nms_boxes = boxes[keep]
        if batch_rots is not None:
          pre_nms_rots = rotation[keep]
        nms_scores = []
        nms_boxes = []
        nms_classes = []
        nms_rots = []
        for class_id in torch.unique(pre_nms_class_ids):
          class_nms_mask = pre_nms_class_ids == class_id
          class_nms_scores = pre_nms_scores[class_nms_mask]
          class_nms_return_scores = pre_nms_return_scores[class_nms_mask]
          class_nms_boxes = pre_nms_boxes[class_nms_mask]
          class_nms_rots = None
          return_rotations = None
          if batch_rots is not None:
            class_nms_rots = pre_nms_rots[class_nms_mask]
            return_rotations = class_nms_rots
            if self.config.normalize_rotation2:
              class_nms_rots = class_nms_rots / 2 + np.pi / 2
          nms_roi, nms_rot, nms_score = utils.non_maximum_suppression(
              class_nms_boxes, class_nms_rots, class_nms_scores,
              self.config.detection_nms_threshold, self.config.detection_max_instances,
              self.config.detection_rot_nms, self.config.detection_aggregate_overlap,
              return_scores=class_nms_return_scores, return_rotations=return_rotations)
          nms_boxes.append(nms_roi)
          nms_scores.append(nms_score)
          nms_classes.append(torch.ones(len(nms_score)).to(class_nms_boxes) * class_id)
          if batch_rots is not None:
            nms_rots.append(nms_rot)
        nms_boxes = torch.cat(nms_boxes)
        nms_scores = torch.cat(nms_scores)
        nms_classes = torch.cat(nms_classes)
        detection_max_instances = min(self.config.detection_max_instances, nms_scores.shape[0])
        ix = torch.topk(nms_scores, detection_max_instances)[1]
        batch_nms.append(torch.cat((nms_boxes[ix], nms_classes[ix, None]), 1))
        batch_nms_scores.append(nms_scores[ix])
        if batch_rots is not None:
          batch_rots.append(torch.cat(nms_rots)[ix])
    return batch_nms, batch_rots, batch_nms_scores

  def get_proposal(self, lrpn_probs, lrpn_sem, ldeltas, lrotation, num_proposals):
    with torch.no_grad():
      assert len(lrpn_probs) == len(lrpn_sem) == len(ldeltas) == len(lrotation)
      rpn_cls = []
      rpn_scores = []
      rpn_return_scores = []
      rpn_boxes = []
      rpn_rotations = []
      rpn_batch_idxs = []
      for rpn_probs, rpn_semantic, deltas, rotation, anchor_size in zip(
              lrpn_probs, lrpn_sem, ldeltas, lrotation, self.anchor_sizes):
        if rpn_probs is None:
          continue
        num_anchors = rpn_probs.F.shape[1] / 2
        assert rpn_probs.coords_key == deltas.coords_key
        assert num_anchors == deltas.F.shape[1] / 6
        rpn_batch_idxs.append(deltas.coords[:, 0])
        rpn_semantic = rpn_semantic.reshape(-1, self.num_class)
        rpn_semantic_prob, rpn_semantic_cls = rpn_semantic.max(1)
        rpn_cls.append(rpn_semantic_cls)
        rpn_prob = rpn_probs.F.reshape(-1, 2)[:, 1]
        if self.config.detection_nms_score == 'obj':
          rpn_score = rpn_prob
        elif self.config.detection_nms_score == 'sem':
          rpn_score = rpn_semantic_prob
        elif self.config.detection_nms_score == 'objsem':
          rpn_score = rpn_prob * rpn_semantic_prob
        if self.config.detection_ap_score == 'obj':
          ap_score = rpn_prob
        elif self.config.detection_ap_score == 'sem':
          ap_score = rpn_semantic_prob
        elif self.config.detection_ap_score == 'objsem':
          ap_score = rpn_prob * rpn_semantic_prob
        rpn_scores.append(rpn_score)
        rpn_return_scores.append(ap_score)
        rpn_bbox_std = torch.from_numpy(np.expand_dims(self.config.rpn_bbox_std, 0)).to(deltas.F)
        anchor_centers = deltas.coords[:, 1:] + deltas.tensor_stride[0] / 2
        anchor_center = np.tile(anchor_centers, (1, int(num_anchors)))
        anchors = np.hstack(((anchor_center - anchor_size).reshape(-1, 3),
                             (anchor_center + anchor_size).reshape(-1, 3)))
        deltas = deltas.F.reshape(-1, 6) * rpn_bbox_std
        anchors = torch.from_numpy(utils.normalize_boxes(anchors, self.config.max_ptc_size))
        rpn_boxes.append(
            utils.apply_box_deltas(anchors.to(deltas), deltas, self.config.normalize_bbox))
        if rotation is not None:
          num_rot_output = self.rotation_criterion.NUM_OUTPUT
          assert rpn_probs.coords_key == rotation.coords_key
          assert rpn_probs.F.shape[1] / 2 == rotation.F.shape[1] / num_rot_output
          rpn_rotations.append(
              self.rotation_criterion.pred(rotation.F.reshape(-1, num_rot_output)))
      if not rpn_scores:
        return None, None, None
      all_scores = torch.cat(rpn_scores)
      all_return_scores = torch.cat(rpn_return_scores)
      all_cls = torch.cat(rpn_cls)
      all_boxes = torch.cat(rpn_boxes)
      all_batch_idxs = torch.cat(rpn_batch_idxs).repeat_interleave(int(num_anchors))
      rotations = None
      if rpn_rotations:
        all_rotations = torch.cat(rpn_rotations)
        rotations = []
      boxes = []
      scores = []
      return_scores = []
      classes = []
      for i in range(all_batch_idxs.max().item() + 1):
        batch_mask = all_batch_idxs == i
        batch_scores = all_scores[batch_mask]
        confidence_mask = batch_scores > self.config.rpn_pre_nms_min_confidence
        pre_nms_limit = min(self.config.rpn_pre_nms_limit, confidence_mask.sum())
        batch_scores, ix = torch.topk(batch_scores[confidence_mask], pre_nms_limit, sorted=True)
        scores.append(batch_scores)
        return_scores.append(all_return_scores[batch_mask][confidence_mask][ix])
        boxes.append(all_boxes[batch_mask][confidence_mask][ix])
        classes.append(all_cls[batch_mask][confidence_mask][ix])
        if rotations is not None:
          rotations.append(all_rotations[batch_mask][confidence_mask][ix])
      rpn_proposal, rotation, rpn_scores = self.batch_non_maximum_suppression(
          boxes, rotations, classes, scores, return_scores, num_proposals)
      return rpn_proposal, rotation, rpn_scores

  def _forward_stride(self, x):
    rpn_class_logits, rpn_probs, rpn_bbox, rpn_rotation = None, None, None, None
    rpn_semantic_logits, rpn_semantic_probs = None, None
    if x is not None:
      x = self.elu(self.bn1(self.conv1(x)))
      feat = self.elu(self.bn2(self.conv2(x)))
      rpn_class_logits = self.final_class_logits(feat)
      rpn_semantic_logits = self.final_semantic_logits(feat)
      num_feat = rpn_semantic_logits.F.shape[0]
      rpn_semantic_probs = F.softmax(rpn_semantic_logits.F.reshape(-1, self.num_class),
                                     dim=1).reshape(num_feat, -1)
      rpn_class_logits_shape = rpn_class_logits.F.shape
      rpn_probs = ME.SparseTensor(
          F.softmax(rpn_class_logits.F.reshape(-1, 2), dim=1).reshape(
              *rpn_class_logits_shape),
          coords_key=rpn_class_logits.coords_key, coords_manager=rpn_class_logits.coords_man)
      rpn_bbox = self.final_bbox(feat)
      if self.is_rotation_bbox:
        rpn_rotation = self.final_rotation(feat)
    return (rpn_class_logits, rpn_probs, rpn_semantic_logits, rpn_semantic_probs, rpn_bbox,
            rpn_rotation)

  def forward(self, fpn_outputs, num_proposals, generate_proposal=True):
    # Forward network for each layer of the pyramid.
    rpn_class_logits, rpn_probs, rpn_semantic_logits, rpn_semantic_probs, rpn_bbox, rpn_rotation \
        = zip(*[self._forward_stride(p) for p in fpn_outputs])
    if generate_proposal:
      # Get bounding boxes from detection output.
      rpn_rois, rpn_rois_rotation, rpn_scores = self.get_proposal(
          rpn_probs, rpn_semantic_probs, rpn_bbox, rpn_rotation, num_proposals)
    else:
      rpn_rois, rpn_rois_rotation, rpn_scores = None, None, None
    return (rpn_class_logits, rpn_semantic_logits, rpn_bbox, rpn_rotation, rpn_rois,
            rpn_rois_rotation, rpn_scores)


class SparsePyramidROIAlign(PyramidROIAlign):
  def _gen_cache(self, x):
    return [sparse_feat.dense() for sparse_feat in x]

  def _get_level_feat(self, x, cache, pyramid_level, batch_idx):
    level_feat, level_min, level_stride = cache[pyramid_level]
    assert level_stride[0] == self.in_pixel_dists[pyramid_level]
    level_feat = level_feat[batch_idx]
    level_min = level_min[0].to(level_feat)
    return level_feat, level_min


class SparseFeaturePyramidClassifierNetwork(FeaturePyramidClassifierNetwork):

  def network_initialization(self, in_channels, config, D):
    super().network_initialization(in_channels, config, D)
    self.roialign = SparsePyramidROIAlign(in_channels, self.in_pixel_dists, self.pool_size, config)
