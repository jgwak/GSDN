from lib.pipelines.detection import FasterRCNN, SparseGenerativeFasterRCNN, \
    SparseGenerativeOneShotDetector, SparseEncoderOnlyOneShotDetector, \
    SparseNoPruningOneShotDetector
from lib.pipelines.instance import MaskRCNN, MaskRCNN_PointNet, MaskRCNN_PointNetXS
from lib.pipelines.segmentation import Segmentation


all_models = [
    FasterRCNN, SparseGenerativeFasterRCNN, SparseGenerativeOneShotDetector,
    SparseEncoderOnlyOneShotDetector, SparseNoPruningOneShotDetector,
    MaskRCNN, MaskRCNN_PointNet, MaskRCNN_PointNetXS,
    Segmentation
]
mdict = {model.__name__: model for model in all_models}


def load_pipeline(config, dataset):
  name = config.pipeline.lower()
  mdict = {model.__name__.lower(): model for model in all_models}
  if name not in mdict:
    print('Invalid pipeline. Options are:')
    # Display a list of valid model names
    for model in all_models:
      print('\t* {}'.format(model.__name__))
    return None
  Class = mdict[name]

  return Class(config, dataset)
