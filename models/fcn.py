import torch
import torch.nn as nn

import MinkowskiEngine as ME

from models.model import Model


class FCNBlocks(nn.Module):

  def __init__(self, feats, pixel_dist, reps, D):
    super(FCNBlocks, self).__init__()

    self.reps = reps
    self.convs, self.bns = {}, {}
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = ME.MinkowskiConvolution(
        feats, feats, pixel_dist=pixel_dist, kernel_size=3, has_bias=False, dimension=D)
    self.bn1 = nn.BatchNorm1d(feats)
    self.conv2 = ME.MinkowskiConvolution(
        feats, feats, pixel_dist=pixel_dist, kernel_size=3, has_bias=False, dimension=D)
    self.bn2 = nn.BatchNorm1d(feats)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    return x


class FCNet(Model):
  """
  FCNet used in the Sparse Conv Net paper. Note that this is different from the
  original FCNet for 2D image segmentation by Long et al.

  dimension = 3
  reps = 2 #Conv block repetition factor
  m = 32 #Unet number of features
  nPlanes = [m, 2*m, 3*m, 4*m, 5*m] #UNet number of features per level

  scn.SubmanifoldConvolution(dimension, 1, m, 3, False)).add(
  scn.FullyConvolutionalNet(
      dimension, reps, nPlanes, residual_blocks=False, downsample=[3,2])).add(
  scn.BatchNormReLU(sum(nPlanes))).add(
  scn.OutputLayer(dimension))

  when residual_blocks=False, use one convolution followed by batchnormrelu.
  """
  OUT_PIXEL_DIST = 1
  INIT = 32
  PLANES = [INIT, 2 * INIT, 3 * INIT, 4 * INIT, 5 * INIT]

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
    super(FCNet, self).__init__(in_channels, out_channels, config, D)
    reps = 2

    # Output of the first conv concated to conv6
    self.conv1p1s1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=self.PLANES[0],
        pixel_dist=1,
        kernel_size=3,
        has_bias=False,
        dimension=D)
    self.bn1 = nn.BatchNorm1d(self.PLANES[0])
    self.block1 = FCNBlocks(self.PLANES[0], pixel_dist=1, reps=reps, D=D)

    self.conv2p1s2 = ME.MinkowskiConvolution(
        in_channels=self.PLANES[0],
        out_channels=self.PLANES[1],
        pixel_dist=1,
        kernel_size=2,
        stride=2,
        has_bias=False,
        dimension=D)
    self.bn2 = nn.BatchNorm1d(self.PLANES[1])
    self.block2 = FCNBlocks(self.PLANES[1], pixel_dist=2, reps=reps, D=D)
    self.unpool2 = ME.MinkowskiPoolingTranspose(pixel_dist=2, kernel_size=2, stride=2, dimension=D)

    self.conv3p2s2 = ME.MinkowskiConvolution(
        in_channels=self.PLANES[1],
        out_channels=self.PLANES[2],
        pixel_dist=2,
        kernel_size=2,
        stride=2,
        has_bias=False,
        dimension=D)
    self.bn3 = nn.BatchNorm1d(self.PLANES[2])
    self.block3 = FCNBlocks(self.PLANES[2], pixel_dist=4, reps=reps, D=D)
    self.unpool3 = ME.MinkowskiPoolingTranspose(pixel_dist=4, kernel_size=4, stride=4, dimension=D)

    self.conv4p4s2 = ME.MinkowskiConvolution(
        in_channels=self.PLANES[2],
        out_channels=self.PLANES[3],
        pixel_dist=4,
        kernel_size=2,
        stride=2,
        has_bias=False,
        dimension=D)
    self.bn4 = nn.BatchNorm1d(self.PLANES[3])
    self.block4 = FCNBlocks(self.PLANES[3], pixel_dist=8, reps=reps, D=D)
    self.unpool4 = ME.MinkowskiPoolingTranspose(pixel_dist=8, kernel_size=8, stride=8, dimension=D)

    self.relu = nn.ReLU(inplace=True)

    self.final = ME.MinkowskiConvolution(
        in_channels=sum(self.PLANES[:4]),
        out_channels=out_channels,
        pixel_dist=1,
        kernel_size=1,
        stride=1,
        has_bias=True,
        dimension=D)

  def forward(self, x):
    out = self.conv1p1s1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out_b1 = self.block1(out)

    out = self.conv2p1s2(out_b1)
    out = self.bn2(out)
    out = self.relu(out)

    out_b2 = self.block2(out)

    out_b2p1 = self.unpool2(out_b2)

    out = self.conv3p2s2(out_b2)
    out = self.bn3(out)
    out = self.relu(out)

    out_b3 = self.block3(out)

    out_b3p1 = self.unpool3(out_b3)

    out = self.conv4p4s2(out_b3)
    out = self.bn4(out)
    out = self.relu(out)

    out_b4 = self.block4(out)

    out_b4p1 = self.unpool4(out_b4)

    out = torch.cat((out_b4p1, out_b3p1, out_b2p1, out_b1), dim=1)
    return self.final(out)
