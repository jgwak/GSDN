from models.modules.senet_block import *

from models.resnet import *
from models.resunet import *
from models.resfcnet import *


class SEResNet14(ResNet14):
  BLOCK = SEBasicBlock


class SEResNet18(ResNet18):
  BLOCK = SEBasicBlock


class SEResNet34(ResNet34):
  BLOCK = SEBasicBlock


class SEResNet50(ResNet50):
  BLOCK = SEBottleneck


class SEResNet101(ResNet101):
  BLOCK = SEBottleneck


class SEResNetIN14(SEResNet14):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = SEBasicBlockIN


class SEResNetIN18(SEResNet18):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = SEBasicBlockIN


class SEResNetIN34(SEResNet34):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = SEBasicBlockIN


class SEResNetIN50(SEResNet50):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = SEBottleneckIN


class SEResNetIN101(SEResNet101):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = SEBottleneckIN


class SEResUNet14(ResUNet14):
  BLOCK = SEBasicBlock


class SEResUNet18(ResUNet18):
  BLOCK = SEBasicBlock


class SEResUNet34(ResUNet34):
  BLOCK = SEBasicBlock


class SEResUNet50(ResUNet50):
  BLOCK = SEBottleneck


class SEResUNet101(ResUNet101):
  BLOCK = SEBottleneck


class SEResUNetIN14(SEResUNet14):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = SEBasicBlockIN


class SEResUNetIN18(SEResUNet18):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = SEBasicBlockIN


class SEResUNetIN34(SEResUNet34):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = SEBasicBlockIN


class SEResUNetIN50(SEResUNet50):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = SEBottleneckIN


class SEResUNetIN101(SEResUNet101):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = SEBottleneckIN


class SEResUNet101(ResUNet101):
  BLOCK = SEBottleneck


class STSEResUNet14(STResUNet14):
  BLOCK = SEBasicBlock


class STSEResUNet18(STResUNet18):
  BLOCK = SEBasicBlock


class STSEResUNet34(STResUNet34):
  BLOCK = SEBasicBlock


class STSEResUNet50(STResUNet50):
  BLOCK = SEBottleneck


class STSEResUNet101(STResUNet101):
  BLOCK = SEBottleneck


class STSEResUNetIN14(STSEResUNet14):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = SEBasicBlockIN


class STSEResUNetIN18(STSEResUNet18):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = SEBasicBlockIN


class STSEResUNetIN34(STResUNet34):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = SEBasicBlockIN


class STSEResUNetIN50(STResUNet50):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = SEBottleneckIN


class STSEResUNetIN101(STResUNet101):
  NORM_TYPE = NormType.SPARSE_INSTANCE_NORM
  BLOCK = SEBottleneckIN


class SEResUNetTemporal14(ResUNetTemporal14):
  BLOCK = SEBasicBlock


class SEResUNetTemporal18(ResUNetTemporal18):
  BLOCK = SEBasicBlock


class SEResUNetTemporal34(ResUNetTemporal34):
  BLOCK = SEBasicBlock


class SEResUNetTemporal50(ResUNetTemporal50):
  BLOCK = SEBottleneck


class STSEResTesseractUNet14(STResTesseractUNet14):
  BLOCK = SEBasicBlock


class STSEResTesseractUNet18(STResTesseractUNet18):
  BLOCK = SEBasicBlock


class STSEResTesseractUNet34(STResTesseractUNet34):
  BLOCK = SEBasicBlock


class STSEResTesseractUNet50(STResTesseractUNet50):
  BLOCK = SEBottleneck


class STSEResTesseractUNet101(STResTesseractUNet101):
  BLOCK = SEBottleneck


class SEResFCNet14(ResFCNet14):
  BLOCK = SEBasicBlock


class SEResFCNet18(ResFCNet18):
  BLOCK = SEBasicBlock


class SEResFCNet34(ResFCNet34):
  BLOCK = SEBasicBlock


class SEResFCNet50(ResFCNet50):
  BLOCK = SEBottleneck


class SEResFCNet101(ResFCNet101):
  BLOCK = SEBottleneck


class STSEResFCNet14(STResFCNet14):
  BLOCK = SEBasicBlock


class STSEResFCNet18(STResFCNet18):
  BLOCK = SEBasicBlock


class STSEResFCNet34(STResFCNet34):
  BLOCK = SEBasicBlock


class STSEResFCNet50(STResFCNet50):
  BLOCK = SEBottleneck


class STSEResFCNet101(STResFCNet101):
  BLOCK = SEBottleneck


class STSEResTesseractFCNet14(STResTesseractFCNet14):
  BLOCK = SEBasicBlock


class STSEResTesseractFCNet18(STResTesseractFCNet18):
  BLOCK = SEBasicBlock


class STSEResTesseractFCNet34(STResTesseractFCNet34):
  BLOCK = SEBasicBlock


class STSEResTesseractFCNet50(STResTesseractFCNet50):
  BLOCK = SEBottleneck


class STSEResTesseractFCNet101(STResTesseractFCNet101):
  BLOCK = SEBottleneck
