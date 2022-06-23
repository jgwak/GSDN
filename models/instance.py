import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME

from models.model import Model
from models.modules.common import conv


class PyramidTrilinearInterpolation(Model):
  def __init__(self, in_channels, in_pixel_dists, config, D=3, **kwargs):
    super().__init__(in_channels, in_channels, config, D, **kwargs)
    self.in_pixel_dists = in_pixel_dists
    self.num_pyramids = len(in_pixel_dists)
    self.OUT_PIXEL_DIST = in_pixel_dists[0]

  def forward(self, batch_rois, batch_coords, x):
    # TODO(jgwak): Incorporate rotation.
    batch_feats_aligned = []
    for batch_idx, (rois, coords) in enumerate(zip(batch_rois, batch_coords)):
      rois_scale = np.cbrt(np.prod(rois[:, 3:6] - rois[:, :3], 1))
      rois_level = np.floor(
          self.config.fpn_base_level + np.log2(rois_scale / self.config.fpn_max_scale))
      rois_level = np.clip(rois_level, 0, self.num_pyramids - 1).astype(int)
      pyramid_idxs_levels = []
      pyramid_feats_levels = []
      for pyramid_level in range(self.num_pyramids):
        pyramid_idxs = np.where(rois_level == pyramid_level)[0]
        if pyramid_idxs.size == 0:
          continue
        pyramid_idxs_levels.append(pyramid_idxs)
        level_feat = x[pyramid_level][batch_idx]
        level_shape = torch.tensor(level_feat.shape[1:]).to(level_feat)
        if self.config.roialign_align_corners:
          level_shape -= 1
        level_feat = level_feat.permute(0, 3, 2, 1)
        level_coords = [coords[i] for i in pyramid_idxs]
        level_numcoords = [c.size(0) for c in level_coords]
        level_grids = torch.cat(level_coords).reshape(1, -1, 1, 1, 3).to(level_feat)
        level_grids /= self.in_pixel_dists[pyramid_level]
        level_grids = level_grids / level_shape * 2 - 1
        coords_feats = F.grid_sample(
            level_feat.unsqueeze(0),
            level_grids,
            align_corners=self.config.roialign_align_corners,
            padding_mode='zeros').reshape(level_feat.shape[0], -1).transpose(0, 1)
        coords_feats = [
            coords_feats[sum(level_numcoords[:i]):sum(level_numcoords[:i + 1])]
            for i in range(len(level_numcoords))
        ]
        pyramid_feats_levels.append(coords_feats)
      if pyramid_feats_levels:
        pyramid_feats = [item for sublist in pyramid_feats_levels for item in sublist]
        pyramid_idxs = np.concatenate(pyramid_idxs_levels).argsort()
        feats_aligned = [pyramid_feats[i] for i in pyramid_idxs]
        batch_feats_aligned.append(feats_aligned)
    batch_coords = [coords.cpu() for subcoords in batch_coords for coords in subcoords]
    if batch_coords:
      batch_coords = ME.utils.batched_coordinates(batch_coords)
      batch_feats = torch.cat([feat for subfeat in batch_feats_aligned for feat in subfeat])
      return ME.SparseTensor(batch_feats, batch_coords)
    else:
      return None


class MaskNetwork(Model):

  def __init__(self, in_channels, config, D=3, **kwargs):
    super().__init__(in_channels, 1, config, D, **kwargs)
    self.mask_feat_size = config.mask_feat_size
    self.network_initialization(in_channels, config, D)
    self.weight_initialization()

  def network_initialization(self, in_channels, config, D):
    self.conv1 = conv(in_channels, self.mask_feat_size, kernel_size=3, stride=1, D=self.D)
    self.bn1 = ME.MinkowskiBatchNorm(self.mask_feat_size, momentum=self.config.bn_momentum)
    self.conv2 = conv(
        self.mask_feat_size, self.mask_feat_size, kernel_size=3, stride=1, D=self.D)
    self.bn2 = ME.MinkowskiBatchNorm(self.mask_feat_size, momentum=self.config.bn_momentum)
    self.conv3 = conv(
        self.mask_feat_size, self.mask_feat_size, kernel_size=3, stride=1, D=self.D)
    self.bn3 = ME.MinkowskiBatchNorm(self.mask_feat_size, momentum=self.config.bn_momentum)
    self.conv4 = conv(
        self.mask_feat_size, self.mask_feat_size, kernel_size=3, stride=1, D=self.D)
    self.bn4 = ME.MinkowskiBatchNorm(self.mask_feat_size, momentum=self.config.bn_momentum)
    self.final = conv(self.mask_feat_size, 1, kernel_size=1, stride=1, D=self.D)
    self.relu = ME.MinkowskiReLU(inplace=True)

  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.relu(self.bn3(self.conv3(x)))
    x = self.relu(self.bn4(self.conv4(x)))
    x = self.final(x)
    return x
