from .scannet import ScannetDataset, Scannet3cmDataset, ScannetVoteNetDataset, \
    ScannetVoteNet3cmDataset, ScannetAlignedDataset, ScannetVoteNetRGBDataset, \
    ScannetVoteNetRGB3cmDataset, ScannetVoteNetRGB25mmDataset
from .synthia import SynthiaDataset
from .sunrgbd import SUNRGBDDataset
from .stanford3d import Stanford3DDataset, Stanford3DSubsampleDataset, \
    Stanford3DMovableObjectsDatasets, Stanford3DMovableObjects3cmDatasets
from .jrdb import JRDataset, JRDataset50, JRDataset30, JRDataset15

DATASETS = [
    ScannetDataset, Scannet3cmDataset, ScannetVoteNetDataset, ScannetVoteNet3cmDataset,
    ScannetVoteNetRGBDataset, ScannetAlignedDataset, SynthiaDataset, SUNRGBDDataset,
    Stanford3DDataset, Stanford3DSubsampleDataset, Stanford3DMovableObjectsDatasets,
    ScannetVoteNetRGB3cmDataset, ScannetVoteNetRGB25mmDataset, Stanford3DMovableObjects3cmDatasets,
    JRDataset, JRDataset50, JRDataset30, JRDataset15
]


def load_dataset(name):
  '''Creates and returns an instance of the datasets given its name.
  '''
  # Find the model class from its name
  mdict = {dataset.__name__: dataset for dataset in DATASETS}
  if name not in mdict:
    print('Invalid dataset index. Options are:')
    # Display a list of valid dataset names
    for dataset in DATASETS:
      print('\t* {}'.format(dataset.__name__))
    return None
  DatasetClass = mdict[name]

  return DatasetClass
