# Change dataloader multiprocess start method to anything not fork
import torch.multiprocessing as mp
try:
  mp.set_start_method('forkserver')  # Reuse process created
except RuntimeError:
  pass

import os
import sys
import logging

# Torch packages
import torch

from config import get_config
from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset
from lib.pipelines import load_pipeline
from lib.test import test
from lib.train import train
from lib.utils import count_parameters

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])


def main():
  config = get_config()

  if config.is_cuda and not torch.cuda.is_available():
    raise Exception("No GPU found")

  # torch.set_num_threads(config.threads)
  torch.manual_seed(config.seed)
  if config.is_cuda:
    torch.cuda.manual_seed(config.seed)

  logging.info('===> Configurations')
  dconfig = vars(config)
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  DatasetClass = load_dataset(config.dataset)

  logging.info('===> Initializing dataloader')
  if config.is_train:
    train_data_loader = initialize_data_loader(
        DatasetClass,
        config,
        phase=config.train_phase,
        threads=config.threads,
        augment_data=True,
        shuffle=True,
        repeat=True,
        batch_size=config.batch_size,
        limit_numpoints=config.train_limit_numpoints)
    val_data_loader = initialize_data_loader(
        DatasetClass,
        config,
        threads=config.val_threads,
        phase=config.val_phase,
        augment_data=False,
        shuffle=False,
        repeat=False,
        batch_size=config.val_batch_size,
        limit_numpoints=False)
    dataset = train_data_loader.dataset
  else:
    test_data_loader = initialize_data_loader(
        DatasetClass,
        config,
        threads=config.threads,
        phase=config.test_phase,
        augment_data=False,
        shuffle=False,
        repeat=False,
        batch_size=config.test_batch_size,
        limit_numpoints=False)
    dataset = test_data_loader.dataset

  logging.info('===> Building model')
  pipeline_model = load_pipeline(config, dataset)
  logging.info(f'===> Number of trainable parameters: {count_parameters(pipeline_model)}')

  # Load weights if specified by the parameter.
  if config.weights.lower() != 'none':
    logging.info('===> Loading weights: ' + config.weights)
    state = torch.load(config.weights)
    pipeline_model.load_state_dict(state['state_dict'], strict=(not config.lenient_weight_loading))
  if config.pretrained_weights.lower() != 'none':
    logging.info('===> Loading pretrained weights: ' + config.pretrained_weights)
    state = torch.load(config.pretrained_weights)
    pipeline_model.load_pretrained_weights(state['state_dict'])

  if config.is_train:
    train(pipeline_model, train_data_loader, val_data_loader, config)
  else:
    test(pipeline_model, test_data_loader, config)


if __name__ == '__main__':
  __spec__ = None
  main()
