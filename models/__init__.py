from models.simplenet import SimpleNet
from models.unet import UNet, UNet2
from models.fcn import FCNet
from models.pointnet import PointNet, PointNetXS
from models.detection import RegionProposalNetwork
from models.instance import MaskNetwork

import models.resnet as resnet
import models.resunet as resunet
import models.res16unet as res16unet
import models.resfcnet as resfcnet
import models.resfuncunet as resfuncunet
import models.senet as senet
import models.resfuncunet as funcunet
import models.segmentation as segmentation

# from models.trilateral_crf import TrilateralCRF
MODELS = [SimpleNet, UNet, UNet2, FCNet, PointNet, PointNetXS, RegionProposalNetwork, MaskNetwork]


def add_models(module, mask='Net'):
  MODELS.extend([getattr(module, a) for a in dir(module) if mask in a])


add_models(resnet)
add_models(resunet)
add_models(res16unet)
add_models(resfcnet)
add_models(resfuncunet)
add_models(senet)
add_models(funcunet)
add_models(segmentation)


def get_models():
  '''Returns a tuple of sample models.'''
  return MODELS


def load_model(name):
  '''Creates and returns an instance of the model given its class name.
  '''
  # Find the model class from its name
  all_models = get_models()
  mdict = {model.__name__: model for model in all_models}
  if name not in mdict:
    print('Invalid model index. Options are:')
    # Display a list of valid model names
    for model in all_models:
      print('\t* {}'.format(model.__name__))
    return None
  NetClass = mdict[name]

  return NetClass
