import torch.nn as nn

import numpy as np
import MinkowskiEngine as ME

from models.model import Model
from models.modules.common import ConvType, NormType, get_norm, conv, sum_pool
from models.modules.resnet_block import BasicBlock, Bottleneck


class ResNetBase(Model):
  BLOCK = None
  LAYERS = ()
  INIT_DIM = 64
  PLANES = (64, 128, 256, 512)
  OUT_PIXEL_DIST = (4, 8, 16, 32)
  NORM_TYPE = NormType.BATCH_NORM
  HAS_LAST_BLOCK = False
  CONV_TYPE = ConvType.HYPERCUBE

  def __init__(self, in_channels, config, D=3, **kwargs):
    assert self.BLOCK is not None

    out_channels = np.array(self.PLANES) * self.BLOCK.expansion
    out_channels = out_channels[-len(self.OUT_PIXEL_DIST):]
    super().__init__(in_channels, out_channels, config, D, **kwargs)

    self.network_initialization(in_channels, config)
    self.weight_initialization()

  def network_initialization(self, in_channels, config):

    dilations = config.dilations
    bn_momentum = config.bn_momentum
    self.inplanes = self.INIT_DIM
    self.conv1 = conv(in_channels, self.inplanes, kernel_size=5, stride=1, D=self.D)

    self.bn1 = get_norm(self.NORM_TYPE, self.inplanes, D=self.D, bn_momentum=bn_momentum)
    self.relu = ME.MinkowskiReLU(inplace=True)
    self.pool = sum_pool(kernel_size=2, stride=2, D=self.D)

    self.layer1 = self._make_layer(
        self.BLOCK,
        self.PLANES[0],
        self.LAYERS[0],
        stride=2,
        norm_type=self.NORM_TYPE,
        dilation=dilations[0])
    self.layer2 = self._make_layer(
        self.BLOCK,
        self.PLANES[1],
        self.LAYERS[1],
        stride=2,
        norm_type=self.NORM_TYPE,
        dilation=dilations[1])
    self.layer3 = self._make_layer(
        self.BLOCK,
        self.PLANES[2],
        self.LAYERS[2],
        stride=2,
        norm_type=self.NORM_TYPE,
        dilation=dilations[2])
    self.layer4 = self._make_layer(
        self.BLOCK,
        self.PLANES[3],
        self.LAYERS[3],
        stride=2,
        norm_type=self.NORM_TYPE,
        dilation=dilations[3])

  def _make_layer(self,
                  block,
                  planes,
                  blocks,
                  stride=1,
                  dilation=1,
                  norm_type=NormType.BATCH_NORM,
                  bn_momentum=0.1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv(
              self.inplanes,
              planes * block.expansion,
              kernel_size=1,
              stride=stride,
              bias=False,
              D=self.D),
          get_norm(norm_type, planes * block.expansion, D=self.D, bn_momentum=bn_momentum),
      )
    layers = []
    layers.append(
        block(
            self.inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            conv_type=self.CONV_TYPE,
            D=self.D))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(
          block(
              self.inplanes,
              planes,
              stride=1,
              dilation=dilation,
              conv_type=self.CONV_TYPE,
              D=self.D))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.pool(x)
    c1 = x = self.layer1(x)
    c2 = x = self.layer2(x)
    c3 = x = self.layer3(x)
    c4 = x = self.layer4(x)

    return [c1, c2, c3, c4]


class ResNet14(ResNetBase):
  BLOCK = BasicBlock
  LAYERS = (1, 1, 1, 1)


class ResNet18(ResNetBase):
  BLOCK = BasicBlock
  LAYERS = (2, 2, 2, 2)


class ResNet34(ResNetBase):
  BLOCK = BasicBlock
  LAYERS = (3, 4, 6, 3)


class ResNet50(ResNetBase):
  BLOCK = Bottleneck
  LAYERS = (3, 4, 6, 3)


class ResNet101(ResNetBase):
  BLOCK = Bottleneck
  LAYERS = (3, 4, 23, 3)


class ResNet14_128(ResNetBase):
  BLOCK = BasicBlock
  PLANES = (128, 128, 256, 512)
  LAYERS = (1, 1, 1, 1)


class ResNet18_128(ResNetBase):
  BLOCK = BasicBlock
  PLANES = (128, 128, 256, 512)
  LAYERS = (2, 2, 2, 2)


class ResNet34_128(ResNetBase):
  BLOCK = BasicBlock
  PLANES = (128, 128, 256, 512)
  LAYERS = (3, 4, 6, 3)


class ResNetHalfBase(ResNetBase):
  INIT_DIM = 64
  PLANES = (64, 64, 128, 256, 512)
  OUT_PIXEL_DIST = (8, 16, 32, 64)

  def network_initialization(self, in_channels, config):

    dilations = config.dilations
    bn_momentum = config.bn_momentum
    self.inplanes = self.INIT_DIM
    self.conv1 = conv(in_channels, self.inplanes, kernel_size=5, stride=1, D=self.D)

    self.bn1 = get_norm(self.NORM_TYPE, self.inplanes, D=self.D, bn_momentum=bn_momentum)
    self.relu = ME.MinkowskiReLU(inplace=True)
    self.pool = sum_pool(kernel_size=2, stride=2, D=self.D)

    self.layer1 = self._make_layer(
        self.BLOCK,
        self.PLANES[0],
        self.LAYERS[0],
        stride=2,
        norm_type=self.NORM_TYPE,
        dilation=dilations[0])
    self.layer2 = self._make_layer(
        self.BLOCK,
        self.PLANES[1],
        self.LAYERS[1],
        stride=2,
        norm_type=self.NORM_TYPE,
        dilation=dilations[1])
    self.layer3 = self._make_layer(
        self.BLOCK,
        self.PLANES[2],
        self.LAYERS[2],
        stride=2,
        norm_type=self.NORM_TYPE,
        dilation=dilations[2])
    self.layer4 = self._make_layer(
        self.BLOCK,
        self.PLANES[3],
        self.LAYERS[3],
        stride=2,
        norm_type=self.NORM_TYPE,
        dilation=dilations[3])
    self.layer5 = self._make_layer(
        self.BLOCK,
        self.PLANES[4],
        self.LAYERS[4],
        stride=2,
        norm_type=self.NORM_TYPE,
        dilation=dilations[4])

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.pool(x)
    x = self.layer1(x)
    c1 = x = self.layer2(x)
    c2 = x = self.layer3(x)
    c3 = x = self.layer4(x)
    c4 = x = self.layer5(x)

    return [c1, c2, c3, c4]


class ResNetHalf15(ResNetHalfBase):
  BLOCK = BasicBlock
  PLANES = (64, 128, 128, 256, 512)
  LAYERS = (1, 1, 1, 1, 1)


class ResNetHalf18(ResNetHalfBase):
  BLOCK = BasicBlock
  PLANES = (64, 128, 128, 256, 512)
  LAYERS = (1, 1, 2, 2, 2)


class ResNetHalf34(ResNetHalfBase):
  BLOCK = BasicBlock
  PLANES = (64, 128, 128, 256, 512)
  LAYERS = (1, 3, 4, 5, 3)


class ResNetHalf50(ResNetHalfBase):
  BLOCK = Bottleneck
  PLANES = (64, 128, 128, 256, 512)
  LAYERS = (1, 3, 4, 5, 3)


class ResNetHalf101(ResNetHalfBase):
  BLOCK = Bottleneck
  PLANES = (64, 128, 128, 256, 512)
  LAYERS = (2, 3, 4, 21, 3)
