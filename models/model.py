from enum import Enum

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from lib.utils import HashTimeBatch


class NetworkType(Enum):
  """
  Classification or segmentation.
  """
  SEGMENTATION = 0, 'SEGMENTATION',
  CLASSIFICATION = 1, 'CLASSIFICATION'

  def __new__(cls, value, name):
    member = object.__new__(cls)
    member._value_ = value
    member.fullname = name
    return member

  def __int__(self):
    return self.value


class Model(ME.MinkowskiNetwork):
  """
  Base network for all sparse convnet

  By default, all networks are segmentation networks.
  """
  OUT_PIXEL_DIST = -1
  NETWORK_TYPE = NetworkType.SEGMENTATION

  def __init__(self, in_channels, out_channels, config, D, **kwargs):
    super(Model, self).__init__(D)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.config = config

  def get_layer(self, layer_name, layer_idx):
    try:
      return self.__getattr__(f'{layer_name}{layer_idx + 1}')
    except AttributeError:
      return None

  def weight_initialization(self):
    for m in self.modules():
      if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, ME.MinkowskiConvolution):
        ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, ME.MinkowskiConvolutionTranspose):
        ME.utils.kaiming_normal_(m.kernel, mode='fan_in', nonlinearity='relu')
      elif isinstance(m, ME.MinkowskiBatchNorm):
        nn.init.constant_(m.bn.weight, 1)
        nn.init.constant_(m.bn.bias, 0)


class SpatialModel(Model):
  """
  Base network for all spatial sparse convnet
  """

  def __init__(self, in_channels, out_channels, config, D, **kwargs):
    assert D == 3, "Num dimension not 3"
    super(SpatialModel, self).__init__(in_channels, out_channels, config, D, **kwargs)

  def initialize_coords(self, coords):
    # In case it has temporal axis
    if coords.size(1) > 4:
      spatial_coord, time, batch = coords[:, :3], coords[:, 3], coords[:, 4]
      time_batch = HashTimeBatch()(time, batch)
      coords = torch.cat((spatial_coord, time_batch.unsqueeze(1)), dim=1)

    super(SpatialModel, self).initialize_coords(coords)


class SpatioTemporalModel(Model):
  """
  Base network for all spatio temporal sparse convnet
  """

  def __init__(self, in_channels, out_channels, config, D=4, **kwargs):
    assert D == 4, "Num dimension not 4"
    super(SpatioTemporalModel, self).__init__(in_channels, out_channels, config, D, **kwargs)


class HighDimensionalModel(Model):
  """
  Base network for all spatio (temporal) chromatic sparse convnet
  """

  def __init__(self, in_channels, out_channels, config, D, **kwargs):
    assert D > 4, "Num dimension smaller than 5"
    super(HighDimensionalModel, self).__init__(in_channels, out_channels, config, D, **kwargs)
