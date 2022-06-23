import MinkowskiEngine as ME

from models.model import Model
from models.modules.common import NormType, get_norm, conv, conv_tr


class SparseFeatureUpsampleNetwork(Model):
  """A sparse network which upsamples and builds a feature pyramid of different strides."""
  NUM_PYRAMIDS = 4

  def __init__(self, in_channels, in_pixel_dists, out_channels, config, D=3, **kwargs):
    assert self.NUM_PYRAMIDS > 0 and config.upsample_feat_size > 0
    assert len(in_channels) == len(in_pixel_dists) == self.NUM_PYRAMIDS
    super().__init__(in_channels, config.upsample_feat_size, config, D, **kwargs)
    self.in_pixel_dists = in_pixel_dists
    self.OUT_PIXEL_DIST = self.in_pixel_dists
    self.network_initialization(in_channels, out_channels, config, D)
    self.weight_initialization()

  def network_initialization(self, in_channels, out_channels, config, D):
    self.conv_feat1 = conv(in_channels[0], config.upsample_feat_size, 3, D=D)
    self.conv_feat2 = conv(in_channels[1], config.upsample_feat_size, 3, D=D)
    self.conv_feat3 = conv(in_channels[2], config.upsample_feat_size, 3, D=D)
    self.conv_feat4 = conv(in_channels[3], config.upsample_feat_size, 3, D=D)
    self.bn_feat1 = get_norm(
        NormType.BATCH_NORM, config.upsample_feat_size, D=3, bn_momentum=config.bn_momentum)
    self.bn_feat2 = get_norm(
        NormType.BATCH_NORM, config.upsample_feat_size, D=3, bn_momentum=config.bn_momentum)
    self.bn_feat3 = get_norm(
        NormType.BATCH_NORM, config.upsample_feat_size, D=3, bn_momentum=config.bn_momentum)
    self.bn_feat4 = get_norm(
        NormType.BATCH_NORM, config.upsample_feat_size, D=3, bn_momentum=config.bn_momentum)
    self.conv_up2 = conv_tr(
        config.upsample_feat_size, config.upsample_feat_size, kernel_size=2, upsample_stride=2,
        dilation=1, bias=False, D=3)
    self.conv_up3 = conv_tr(
        config.upsample_feat_size, config.upsample_feat_size, kernel_size=2, upsample_stride=2,
        dilation=1, bias=False, D=3)
    self.conv_up4 = conv_tr(
        config.upsample_feat_size, config.upsample_feat_size, kernel_size=2, upsample_stride=2,
        dilation=1, bias=False, D=3)
    self.conv_up5 = conv_tr(
        config.upsample_feat_size, config.upsample_feat_size, kernel_size=2, upsample_stride=2,
        dilation=1, bias=False, D=3)
    self.conv_up6 = conv_tr(
        config.upsample_feat_size, config.upsample_feat_size, kernel_size=2, upsample_stride=2,
        dilation=1, bias=False, D=3)
    self.bn_up2 = get_norm(
        NormType.BATCH_NORM, config.upsample_feat_size, D=3, bn_momentum=config.bn_momentum)
    self.bn_up3 = get_norm(
        NormType.BATCH_NORM, config.upsample_feat_size, D=3, bn_momentum=config.bn_momentum)
    self.bn_up4 = get_norm(
        NormType.BATCH_NORM, config.upsample_feat_size, D=3, bn_momentum=config.bn_momentum)
    self.bn_up5 = get_norm(
        NormType.BATCH_NORM, config.upsample_feat_size, D=3, bn_momentum=config.bn_momentum)
    self.bn_up6 = get_norm(
        NormType.BATCH_NORM, config.upsample_feat_size, D=3, bn_momentum=config.bn_momentum)
    self.conv_final = conv(config.upsample_feat_size, out_channels, 1, D=D)
    self.relu = ME.MinkowskiReLU(inplace=False)

  def forward(self, backbone_outputs):
    pyramid_output = None
    for layer_idx in reversed(range(len(backbone_outputs))):
      sparse_tensor = backbone_outputs[layer_idx]
      conv_feat = self.get_layer('conv_feat', layer_idx)
      bn_feat = self.get_layer('bn_feat', layer_idx)
      fpn_feat = self.relu(bn_feat(conv_feat(sparse_tensor)))
      if pyramid_output is not None:
        fpn_feat += pyramid_output
      conv_up = self.get_layer('conv_up', layer_idx)
      if conv_up is not None:
        bn_up = self.get_layer('bn_up', layer_idx)
        pyramid_output = self.relu(bn_up(conv_up(fpn_feat)))
    fpn_feat = self.relu(self.bn_up5(self.conv_up5(fpn_feat)))
    fpn_feat = self.relu(self.bn_up6(self.conv_up6(fpn_feat)))
    fpn_output = self.conv_final(fpn_feat)
    return fpn_output
