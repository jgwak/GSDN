import collections
import logging
import os
import tempfile

import torch

from lib.utils import Timer, AverageMeter, log_meters


def test(pipeline_model, data_loader, config, has_gt=True):
  global_timer, data_timer, iter_timer = Timer(), Timer(), Timer()
  meters = collections.defaultdict(AverageMeter)
  hists = pipeline_model.initialize_hists()

  logging.info('===> Start testing')

  global_timer.tic()
  data_iter = data_loader.__iter__()
  max_iter = len(data_loader)

  # Fix batch normalization running mean and std
  pipeline_model.eval()

  # Clear cache (when run in val mode, cleanup training cache)
  torch.cuda.empty_cache()

  if config.save_prediction or config.test_original_pointcloud:
    if config.save_prediction:
      save_pred_dir = config.save_pred_dir
      os.makedirs(save_pred_dir, exist_ok=True)
    else:
      save_pred_dir = tempfile.mkdtemp()
    if os.listdir(save_pred_dir):
      raise ValueError(f'Directory {save_pred_dir} not empty. '
                       'Please remove the existing prediction.')

  with torch.no_grad():
    for iteration in range(max_iter):
      iter_timer.tic()
      data_timer.tic()
      datum = pipeline_model.load_datum(data_iter, has_gt=has_gt)
      data_time = data_timer.toc(False)

      output_dict = pipeline_model(datum, False)
      iter_time = iter_timer.toc(False)

      if config.save_prediction or config.test_original_pointcloud:
        pipeline_model.save_prediction(datum, output_dict, save_pred_dir, iteration)

      if config.visualize and iteration % config.visualize_freq == 0:
        pipeline_model.visualize_predictions(datum, output_dict, iteration)

      if has_gt:
        loss_dict = pipeline_model.loss(datum, output_dict)
        if config.visualize and iteration % config.visualize_freq == 0:
          pipeline_model.visualize_groundtruth(datum, iteration)
        loss_dict.update(pipeline_model.evaluate(datum, output_dict))

        meters, hists = pipeline_model.update_meters(meters, hists, loss_dict)

      if iteration % config.test_stat_freq == 0 and iteration > 0:
        debug_str = "===> {}/{}\n".format(iteration, max_iter)
        debug_str += log_meters(meters, log_perclass_meters=True)
        debug_str += f"\n    data time: {data_time:.3f}    iter time: {iter_time:.3f}"
        logging.info(debug_str)

      if iteration % config.empty_cache_freq == 0:
        # Clear cache
        torch.cuda.empty_cache()

  global_time = global_timer.toc(False)

  debug_str = "===> Final test results:\n"
  debug_str += log_meters(meters, log_perclass_meters=True)
  logging.info(debug_str)

  if config.test_original_pointcloud:
    pipeline_model.test_original_pointcloud(save_pred_dir)

  logging.info('Finished test. Elapsed time: {:.4f}'.format(global_time))

  # Explicit memory cleanup
  if hasattr(data_iter, 'cleanup'):
    data_iter.cleanup()

  return meters
