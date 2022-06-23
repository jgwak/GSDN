import torch
import torch.nn as nn

from models.resnet import ResNetBase, ResNet14, ResNet18, ResNet34, ResNet50, ResNet101
from models.modules.common import ConvType, conv, conv_tr


class ResFCNetBase(ResNetBase):
  OUT_PIXEL_DIST = 1

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
    super(ResFCNetBase, self).__init__(in_channels, out_channels, config, D)

  def network_initialization(self, in_channels, out_channels, config, D):
    net_metadata = self.net_metadata

    def space_n_time_m(n, m):
      return n if D == 3 else [n, n, n, m]

    if D == 4:
      self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

    # Output of the first conv concated to conv6
    self.inplanes = self.PLANES[0]
    self.conv1 = conv(
        in_channels,
        self.inplanes,
        pixel_dist=1,
        kernel_size=space_n_time_m(5, 1),
        stride=1,
        dilation=1,
        bias=False,
        D=D,
        net_metadata=net_metadata)

    self.bn1 = nn.BatchNorm1d(self.inplanes)
    self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0], pixel_dist=1)

    self.conv2p1s2 = conv(
        self.inplanes,
        self.inplanes,
        pixel_dist=1,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS,
        bias=False,
        D=D,
        net_metadata=net_metadata)
    self.bn2 = nn.BatchNorm1d(self.inplanes)
    self.block2 = self._make_layer(
        self.BLOCK, self.PLANES[1], self.LAYERS[1], pixel_dist=space_n_time_m(2, 1))
    self.convtr2p2s2 = conv_tr(
        self.inplanes,
        self.PLANES[1],
        pixel_dist=space_n_time_m(2, 1),
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS,
        bias=False,
        D=D,
        net_metadata=net_metadata)
    self.bntr2 = nn.BatchNorm1d(self.PLANES[1])

    self.conv3p2s2 = conv(
        self.inplanes,
        self.inplanes,
        pixel_dist=space_n_time_m(2, 1),
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS,
        bias=False,
        D=D,
        net_metadata=net_metadata)
    self.bn3 = nn.BatchNorm1d(self.inplanes)
    self.block3 = self._make_layer(
        self.BLOCK, self.PLANES[2], self.LAYERS[2], pixel_dist=space_n_time_m(4, 1))

    self.convtr3p4s4 = conv_tr(
        self.inplanes,
        self.PLANES[2],
        pixel_dist=space_n_time_m(4, 1),
        kernel_size=space_n_time_m(4, 1),
        upsample_stride=space_n_time_m(4, 1),
        dilation=1,
        conv_type=ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS,
        bias=False,
        D=D,
        net_metadata=net_metadata)
    self.bntr3 = nn.BatchNorm1d(self.PLANES[2])

    self.conv4p4s2 = conv(
        self.inplanes,
        self.inplanes,
        pixel_dist=space_n_time_m(4, 1),
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS,
        bias=False,
        D=D,
        net_metadata=net_metadata)
    self.bn4 = nn.BatchNorm1d(self.inplanes)
    self.block4 = self._make_layer(
        self.BLOCK, self.PLANES[3], self.LAYERS[3], pixel_dist=space_n_time_m(8, 1))
    self.convtr4p8s8 = conv_tr(
        self.inplanes,
        self.PLANES[3],
        pixel_dist=space_n_time_m(8, 1),
        kernel_size=space_n_time_m(8, 1),
        upsample_stride=space_n_time_m(8, 1),
        dilation=1,
        conv_type=ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS,
        bias=False,
        D=D,
        net_metadata=net_metadata)
    self.bntr4 = nn.BatchNorm1d(self.PLANES[3])

    self.relu = nn.ReLU(inplace=True)

    self.final = conv(
        sum(self.PLANES[1:4]) + self.PLANES[0] * self.BLOCK.expansion,
        out_channels,
        pixel_dist=1,
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=True,
        D=D,
        net_metadata=net_metadata)

  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out_b1 = self.block1(out)

    out = self.conv2p1s2(out_b1)
    out = self.bn2(out)
    out = self.relu(out)

    out_b2 = self.block2(out)

    out = self.convtr2p2s2(out_b2)
    out = self.bntr2(out)
    out_b2p1 = self.relu(out)

    out = self.conv3p2s2(out_b2)
    out = self.bn3(out)
    out = self.relu(out)

    out_b3 = self.block3(out)

    out = self.convtr3p4s4(out_b3)
    out = self.bntr3(out)
    out_b3p1 = self.relu(out)

    out = self.conv4p4s2(out_b3)
    out = self.bn4(out)
    out = self.relu(out)

    out_b4 = self.block4(out)

    out = self.convtr4p8s8(out_b4)
    out = self.bntr4(out)
    out_b4p1 = self.relu(out)

    if self.USE_VALID_CONV:
      out_b1 = self.unpool1(out_b1)

    out = torch.cat((out_b4p1, out_b3p1, out_b2p1, out_b1), dim=1)
    return self.final(out)


class ResFCNet14(ResFCNetBase, ResNet14):
  pass


class ResFCNet18(ResFCNetBase, ResNet18):
  pass


class ResFCNet34(ResFCNetBase, ResNet34):
  pass


class ResFCNet50(ResFCNetBase, ResNet50):
  PLANES = (64, 128, 256, 384)


class ResFCNet101(ResFCNetBase, ResNet101):
  PLANES = (64, 128, 256, 384)


class STResFCNetBase(ResFCNetBase):

  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

  def __init__(self, in_channels, out_channels, config, D=4, **kwargs):
    super(STResFCNetBase, self).__init__(in_channels, out_channels, config, D, **kwargs)


class STResFCNet14(STResFCNetBase, ResFCNet14):
  pass


class STResFCNet18(STResFCNetBase, ResFCNet18):
  pass


class STResFCNet34(STResFCNetBase, ResFCNet34):
  pass


class STResFCNet50(STResFCNetBase, ResFCNet50):
  pass


class STResFCNet101(STResFCNetBase, ResFCNet101):
  pass


class STResTesseractFCNetBase(STResFCNetBase):
  CONV_TYPE = ConvType.HYPERCUBE


class STResTesseractFCNet14(STResTesseractFCNetBase, ResFCNet14):
  pass


class STResTesseractFCNet18(STResTesseractFCNetBase, ResFCNet18):
  pass


class STResTesseractFCNet34(STResTesseractFCNetBase, ResFCNet34):
  pass


class STResTesseractFCNet50(STResTesseractFCNetBase, ResFCNet50):
  pass


class STResTesseractFCNet101(STResTesseractFCNetBase, ResFCNet101):
  pass
