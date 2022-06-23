import torch.nn as nn

from models.model import Model
from models.resnet import get_norm
from models.modules.common import ConvType, NormType, conv, conv_tr
from models.modules.resnet_block import BasicBlock

from MinkowskiEngine import MinkowskiReLU, MinkowskiOps


def space_n_time_m(n, m, D):
  return n if D == 3 else [n, n, n, m]


class UBlock(nn.Module):

  def __init__(self,
               inplanes,
               intermediate_inplanes,
               intermediate_outplanes,
               outplanes,
               intermediate_module,
               BLOCK=None,
               reps=1,
               conv_type=ConvType.HYPERCUBE,
               norm_type=NormType.BATCH_NORM,
               bn_momentum=0.1,
               D=3):
    super(UBlock, self).__init__()

    self.block = BLOCK(inplanes, inplanes, conv_type=conv_type, bn_momentum=bn_momentum, D=D)
    self.down = conv(
        inplanes,
        intermediate_inplanes,
        kernel_size=space_n_time_m(2, 1, D),
        stride=space_n_time_m(2, 1, D),
        conv_type=conv_type,
        D=D)
    self.down_norm = get_norm(norm_type, intermediate_inplanes, D, bn_momentum=bn_momentum)
    self.intermediate = intermediate_module
    self.up = conv_tr(
        intermediate_outplanes,
        outplanes,
        kernel_size=space_n_time_m(2, 1, D),
        upsample_stride=space_n_time_m(2, 1, D),
        conv_type=conv_type,
        D=D)
    self.up_norm = get_norm(norm_type, outplanes, D, bn_momentum=bn_momentum)
    self.reps = reps
    for i in range(reps):
      if i == 0:
        downsample = nn.Sequential(
            conv(inplanes + outplanes, outplanes, kernel_size=1, bias=False, D=D),
            get_norm(norm_type, outplanes, D, bn_momentum=bn_momentum),
        )

      setattr(
          self, f'end_blocks{i}',
          BLOCK(
              (inplanes + outplanes) if i == 0 else outplanes,
              outplanes,
              downsample=downsample if i == 0 else None,
              conv_type=conv_type,
              bn_momentum=bn_momentum,
              D=D))

  def forward(self, x):
    out = self.block(x)
    out = self.down(out)
    out = self.down_norm(out)
    out = self.intermediate(out)
    out = self.up(out)
    out = self.up_norm(out)
    out = MinkowskiOps.cat((out, x))
    for i in range(self.reps):
      out = getattr(self, f'end_blocks{i}')(out)
    return out


class RecUNetBase(Model):
  BLOCK = None
  INIT_DIM = 32
  PLANES = ([INIT_DIM, 4 * INIT_DIM], [2 * INIT_DIM, 4 * INIT_DIM], [3 * INIT_DIM, 4 * INIT_DIM],
            [4 * INIT_DIM, 4 * INIT_DIM], [5 * INIT_DIM, 5 * INIT_DIM],
            [6 * INIT_DIM, 6 * INIT_DIM], [7 * INIT_DIM, 7 * INIT_DIM])
  REPS = (1, 1, 1, 1, 1, 1, 1)
  OUT_PIXEL_DIST = 1
  NORM_TYPE = NormType.BATCH_NORM
  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
    super(RecUNetBase, self).__init__(in_channels, out_channels, config, D)

    PLANES = self.PLANES[::-1]
    bn_momentum = config.bn_momentum

    if D == 4:
      self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1, D)

    # Output of the first conv concated to conv6
    self.conv1 = conv(
        in_channels,
        PLANES[-1][0],
        kernel_size=space_n_time_m(3, 1, D),
        conv_type=self.CONV_TYPE,
        D=D)
    self.norm1 = get_norm(self.NORM_TYPE, PLANES[-1][0], D, bn_momentum)
    interm = self.BLOCK(PLANES[0][0], PLANES[0][0], conv_type=self.CONV_TYPE, D=self.D)

    for i, inoutplanes in enumerate(PLANES[1:]):
      interm = UBlock(
          inoutplanes[0],
          PLANES[i][0],
          PLANES[i][1],
          inoutplanes[1],
          intermediate_module=interm,
          BLOCK=self.BLOCK,
          reps=self.REPS[len(self.REPS) - i - 1],
          conv_type=self.CONV_TYPE,
          bn_momentum=bn_momentum,
          D=D)
    self.unet = interm
    self.final = conv(
        PLANES[-1][1], out_channels, kernel_size=1, stride=1, dilation=1, bias=True, D=D)

    self.relu = MinkowskiReLU(inplace=True)

  def forward(self, x):
    out = self.conv1(x)
    out = self.norm1(out)
    out_b1p1 = self.relu(out)
    out = self.unet(out_b1p1)
    return self.final(out)


class RecUNet45(RecUNetBase):
  BLOCK = BasicBlock


class RecUNet60(RecUNetBase):
  BLOCK = BasicBlock
  REPS = (2, 2, 2, 2, 2, 2, 2)


class RecUNet45A(RecUNet45):
  INIT_DIM = 32
  PLANES = ([INIT_DIM, 3 * INIT_DIM], [2 * INIT_DIM, 3 * INIT_DIM], [3 * INIT_DIM, 3 * INIT_DIM],
            [4 * INIT_DIM, 4 * INIT_DIM], [5 * INIT_DIM, 5 * INIT_DIM],
            [6 * INIT_DIM, 6 * INIT_DIM], [7 * INIT_DIM, 7 * INIT_DIM])


class RecUNet45B(RecUNet45):
  INIT_DIM = 32
  PLANES = ([INIT_DIM, 2 * INIT_DIM], [2 * INIT_DIM, 2 * INIT_DIM], [3 * INIT_DIM, 3 * INIT_DIM],
            [4 * INIT_DIM, 4 * INIT_DIM], [5 * INIT_DIM, 5 * INIT_DIM],
            [6 * INIT_DIM, 6 * INIT_DIM], [7 * INIT_DIM, 7 * INIT_DIM])


class RecUNet45C(RecUNet45):
  INIT_DIM = 16


class RecUNet45D(RecUNet45):
  INIT_DIM = 24


class RecUNetInst45(RecUNet45):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM


class RecUNetInst45A(RecUNet45A):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM


class RecUNetInst45B(RecUNet45B):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM


class RecUNetInst45C(RecUNet45C):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM


class RecUNetInst45D(RecUNet45D):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM


class RecUNet60B(RecUNet60):
  INIT_DIM = 32
  PLANES = ([INIT_DIM, 2 * INIT_DIM], [2 * INIT_DIM, 2 * INIT_DIM], [3 * INIT_DIM, 3 * INIT_DIM],
            [4 * INIT_DIM, 4 * INIT_DIM], [5 * INIT_DIM, 5 * INIT_DIM],
            [6 * INIT_DIM, 6 * INIT_DIM], [7 * INIT_DIM, 7 * INIT_DIM])


class RecUNetInst60B(RecUNet60B):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
