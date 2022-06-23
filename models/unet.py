import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me

from models.model import Model
from models.modules.common import conv, conv_tr


class UNBlocks(nn.Module):
  """ Replaces two blocks of the UNet block of SparseConvNet into one."""

  def __init__(self, in_feats, out_feats, D):
    super(UNBlocks, self).__init__()

    self.convs, self.bns = {}, {}
    self.relu = ME.MinkowskiReLU(inplace=True)
    self.conv1 = conv(in_feats, out_feats, kernel_size=3, bias=False, D=D)
    self.bn1 = ME.MinkowskiBatchNorm(out_feats)
    self.conv2 = conv(out_feats, out_feats, kernel_size=3, bias=False, D=D)
    self.bn2 = ME.MinkowskiBatchNorm(out_feats)

  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn2(self.conv2(x)))
    return x


class UNet2(Model):
  """
  reps = 2
  m = 32 #Unet number of features
  nPlanes = [m, 2*m, 3*m, 4*m, 5*m] #UNet number of features per level

  self.sparseModel = scn.Sequential().add(
     scn.InputLayer(dimension, data.spatialSize, mode=3)).add(
     scn.SubmanifoldConvolution(dimension, 1, m, 3, False)).add(
     scn.UNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[3,2])).add(
     scn.BatchNormReLU(m)).add(
     scn.OutputLayer(dimension))
  self.linear = nn.Linear(m, data.nClassesTotal)
  """
  OUT_PIXEL_DIST = 1
  INIT = 64
  PLANES = [INIT, 2 * INIT, 4 * INIT, 2 * INIT, 1 * INIT]

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
    super(UNet2, self).__init__(in_channels, out_channels, config, D)
    PLANES = self.PLANES

    # Output of the first conv concated to conv6
    self.conv1 = conv(in_channels, PLANES[0], kernel_size=3, stride=1, bias=False, D=D)
    self.bn1 = ME.MinkowskiBatchNorm(PLANES[0])

    self.block1 = UNBlocks(PLANES[0], PLANES[0], D)
    self.down1 = conv(PLANES[0], PLANES[1], kernel_size=2, stride=2, D=D)
    self.down1bn = ME.MinkowskiBatchNorm(PLANES[1])

    self.up1 = conv_tr(PLANES[1], PLANES[0], kernel_size=2, upsample_stride=2, D=D)
    self.block1up = UNBlocks(PLANES[0] * 2, PLANES[0], D)

    self.block2 = UNBlocks(PLANES[1], PLANES[1], D)
    self.down2 = conv(PLANES[1], PLANES[2], kernel_size=2, stride=2, D=D)

    self.up2 = conv_tr(PLANES[2], PLANES[1], kernel_size=2, upsample_stride=2, D=D)
    self.block2up = UNBlocks(PLANES[1] * 2, PLANES[1], D)

    self.block3 = UNBlocks(PLANES[2], PLANES[2], D)
    self.down3 = conv(PLANES[2], PLANES[3], kernel_size=2, stride=2, D=D)

    self.up3 = conv_tr(PLANES[3], PLANES[2], kernel_size=2, upsample_stride=2, D=D)
    self.block3up = UNBlocks(PLANES[2] * 2, PLANES[2], D)

    self.block4 = UNBlocks(PLANES[3], PLANES[3], D)

    self.relu = ME.MinkowskiReLU(inplace=True)
    self.final = conv(PLANES[0], out_channels, kernel_size=1, bias=True, D=D)

  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out_b1 = self.block1(out)
    out = self.down1(out_b1)

    out_b2 = self.block2(out)
    out = self.down2(out_b2)

    out_b3 = self.block3(out)
    out = self.down3(out_b3)

    out = self.block4(out)
    out = self.up3(out)

    out = self.block3up(me.cat((out_b3, out)))

    out = self.up2(out)
    out = self.block2up(me.cat((out_b2, out)))

    out = self.up1(out)
    out = self.block1up(me.cat((out_b1, out)))

    return self.final(out)


class UNet(Model):
  """
  reps = 2
  m = 32 #Unet number of features
  nPlanes = [m, 2*m, 3*m, 4*m, 5*m] #UNet number of features per level

  self.sparseModel = scn.Sequential().add(
     scn.InputLayer(dimension, data.spatialSize, mode=3)).add(
     scn.SubmanifoldConvolution(dimension, 1, m, 3, False)).add(
     scn.UNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[3,2])).add(
     scn.BatchNormReLU(m)).add(
     scn.OutputLayer(dimension))
  self.linear = nn.Linear(m, data.nClassesTotal)
  """
  OUT_PIXEL_DIST = 1
  INIT = 64
  PLANES = [INIT, 2 * INIT, 4 * INIT, 4 * INIT, 4 * INIT, 2 * INIT, INIT]

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self, in_channels, out_channels, config, D=3, return_feat=False, **kwargs):
    super(UNet, self).__init__(in_channels, out_channels, config, D)
    self.in_channels = in_channels
    self.return_feat = return_feat
    PLANES = self.PLANES

    # Output of the first conv concated to conv6
    self.conv_down1 = conv(in_channels, PLANES[0], kernel_size=3, stride=1, bias=False, D=D)
    self.bn_down1 = ME.MinkowskiBatchNorm(PLANES[0])
    self.down1 = conv(PLANES[0], PLANES[1], kernel_size=2, stride=2, D=D)
    self.conv_down2 = conv(PLANES[1], PLANES[1], kernel_size=3, stride=1, bias=False, D=D)
    self.bn_down2 = ME.MinkowskiBatchNorm(PLANES[1])
    self.down2 = conv(PLANES[1], PLANES[2], kernel_size=2, stride=2, D=D)
    self.conv_down3 = conv(PLANES[2], PLANES[2], kernel_size=3, stride=1, bias=False, D=D)
    self.bn_down3 = ME.MinkowskiBatchNorm(PLANES[2])
    self.down3 = conv(PLANES[2], PLANES[3], kernel_size=2, stride=2, D=D)
    self.conv_down4 = conv(PLANES[3], PLANES[3], kernel_size=3, stride=1, bias=False, D=D)
    self.bn_down4 = ME.MinkowskiBatchNorm(PLANES[3])
    self.down4 = conv(PLANES[3], PLANES[4], kernel_size=2, stride=2, D=D)
    self.conv_down5 = conv(PLANES[4], PLANES[4], kernel_size=3, stride=1, bias=False, D=D)
    self.bn_down5 = ME.MinkowskiBatchNorm(PLANES[4])
    self.down5 = conv(PLANES[4], PLANES[5], kernel_size=2, stride=2, D=D)
    self.conv_down6 = conv(PLANES[5], PLANES[5], kernel_size=3, stride=1, bias=False, D=D)
    self.bn_down6 = ME.MinkowskiBatchNorm(PLANES[5])
    self.down6 = conv(PLANES[5], PLANES[6], kernel_size=2, stride=2, D=D)
    self.conv7 = conv(PLANES[6], PLANES[6], kernel_size=3, stride=1, bias=False, D=D)
    self.bn7 = ME.MinkowskiBatchNorm(PLANES[6])
    self.up6 = conv_tr(PLANES[6], PLANES[5], kernel_size=2, upsample_stride=2, D=D)
    self.conv_up6 = conv(PLANES[5] * 2, PLANES[5], kernel_size=3, stride=1, bias=False, D=D)
    self.bn_up6 = ME.MinkowskiBatchNorm(PLANES[5])
    self.up5 = conv_tr(PLANES[5], PLANES[4], kernel_size=2, upsample_stride=2, D=D)
    self.conv_up5 = conv(PLANES[4] * 2, PLANES[4], kernel_size=3, stride=1, bias=False, D=D)
    self.bn_up5 = ME.MinkowskiBatchNorm(PLANES[4])
    self.up4 = conv_tr(PLANES[4], PLANES[3], kernel_size=2, upsample_stride=2, D=D)
    self.conv_up4 = conv(PLANES[3] * 2, PLANES[3], kernel_size=3, stride=1, bias=False, D=D)
    self.bn_up4 = ME.MinkowskiBatchNorm(PLANES[3])
    self.up3 = conv_tr(PLANES[3], PLANES[2], kernel_size=2, upsample_stride=2, D=D)
    self.conv_up3 = conv(PLANES[2] * 2, PLANES[2], kernel_size=3, stride=1, bias=False, D=D)
    self.bn_up3 = ME.MinkowskiBatchNorm(PLANES[2])
    self.up2 = conv_tr(PLANES[2], PLANES[1], kernel_size=2, upsample_stride=2, D=D)
    self.conv_up2 = conv(PLANES[1] * 2, PLANES[1], kernel_size=3, stride=1, bias=False, D=D)
    self.bn_up2 = ME.MinkowskiBatchNorm(PLANES[1])
    self.up1 = conv_tr(PLANES[1], PLANES[0], kernel_size=2, upsample_stride=2, D=D)
    self.conv_up1 = conv(PLANES[0] * 2, PLANES[0], kernel_size=3, stride=1, bias=False, D=D)
    self.bn_up1 = ME.MinkowskiBatchNorm(PLANES[0])
    self.mask_feat = conv(PLANES[0], self.in_channels, kernel_size=1, bias=True, D=D)
    self.final = conv(PLANES[0], out_channels, kernel_size=1, bias=True, D=D)
    self.relu = ME.MinkowskiReLU(inplace=True)

  def forward(self, x):
    out_b1 = self.relu(self.bn_down1(self.conv_down1(x)))
    out = self.down1(out_b1)
    out_b2 = self.relu(self.bn_down2(self.conv_down2(out)))
    out = self.down2(out_b2)
    out_b3 = self.relu(self.bn_down3(self.conv_down3(out)))
    out = self.down3(out_b3)
    out_b4 = self.relu(self.bn_down4(self.conv_down4(out)))
    out = self.down4(out_b4)
    out_b5 = self.relu(self.bn_down5(self.conv_down5(out)))
    out = self.down5(out_b5)
    out_b6 = self.relu(self.bn_down6(self.conv_down6(out)))
    out = self.down6(out_b6)
    out = self.relu(self.bn7(self.conv7(out)))
    out = self.up6(out)
    out = self.relu(self.bn_up6(self.conv_up6(me.cat((out_b6, out)))))
    out = self.up5(out)
    out = self.relu(self.bn_up5(self.conv_up5(me.cat((out_b5, out)))))
    out = self.up4(out)
    out = self.relu(self.bn_up4(self.conv_up4(me.cat((out_b4, out)))))
    out = self.up3(out)
    out = self.relu(self.bn_up3(self.conv_up3(me.cat((out_b3, out)))))
    out = self.up2(out)
    out = self.relu(self.bn_up2(self.conv_up2(me.cat((out_b2, out)))))
    out = self.up1(out)
    out_feat = self.relu(self.bn_up1(self.conv_up1(me.cat((out_b1, out)))))
    out = self.final(out_feat)

    if self.return_feat:
      feat = self.mask_feat(out_feat)
      return feat, out

    return out
