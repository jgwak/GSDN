import torch.nn as nn

import MinkowskiEngine as ME

from models.model import Model


class SimpleNet(Model):
  OUT_PIXEL_DIST = 4

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
    super(SimpleNet, self).__init__(in_channels, out_channels, config, D)
    kernel_size = 3
    self.conv1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=64,
        pixel_dist=1,
        kernel_size=kernel_size,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.bn1 = nn.BatchNorm1d(64)

    self.conv2 = ME.MinkowskiConvolution(
        in_channels=64,
        out_channels=128,
        pixel_dist=2,
        kernel_size=kernel_size,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.bn2 = nn.BatchNorm1d(128)

    self.conv3 = ME.MinkowskiConvolution(
        in_channels=128,
        out_channels=128,
        pixel_dist=4,
        kernel_size=kernel_size,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.bn3 = nn.BatchNorm1d(128)

    self.conv4 = ME.MinkowskiConvolution(
        in_channels=128,
        out_channels=128,
        pixel_dist=4,
        kernel_size=kernel_size,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.bn4 = nn.BatchNorm1d(128)

    self.conv5 = ME.MinkowskiConvolution(
        in_channels=128,
        out_channels=out_channels,
        pixel_dist=4,
        kernel_size=kernel_size,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.bn5 = nn.BatchNorm1d(out_channels)

    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)
    out = self.relu(out)

    out = self.conv4(out)
    out = self.bn4(out)
    out = self.relu(out)

    return self.conv5(out)
