import collections
import logging
import os.path as osp

import torch
import MinkowskiEngine as ME
import torch.nn.parallel as parallel

from tensorboardX import SummaryWriter

from lib.test import test
from lib.utils import checkpoint, Timer, AverageMeter, update_writer, log_meters, reset_meters, \
    unconvert_sync_batchnorm


def validate(pipeline_model, data_loader, config, writer, curr_iter, best_val, best_val_iter,
             optimizer, epoch):
  val_dict = test(pipeline_model, data_loader, config)
  update_writer(writer, val_dict, curr_iter, 'validation')
  curr_val = pipeline_model.get_metric(val_dict)

  if curr_val > best_val:
    best_val = curr_val
    best_val_iter = curr_iter
    checkpoint(
        pipeline_model, optimizer, epoch, curr_iter, config, best_val, best_val_iter, 'best_val')
  logging.info(
      f'Current best {pipeline_model.TARGET_METRIC}: {best_val:.3f} at iter {best_val_iter}')

  # Recover back
  pipeline_model.train()

  return best_val, best_val_iter


def train(pipeline_model, data_loader, val_data_loader, config):
  # Set up the train flag for batch normalization
  pipeline_model.train()

  num_devices = torch.cuda.device_count()
  num_devices = min(config.max_ngpu, num_devices)
  devices = list(range(num_devices))
  target_device = devices[0]
  pipeline_model.to(target_device)
  if num_devices > 1:
    pipeline_model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(pipeline_model, devices)

  # Configuration
  writer = SummaryWriter(logdir=config.log_dir)
  data_timer, iter_timer = Timer(), Timer()
  data_time_avg, iter_time_avg = AverageMeter(), AverageMeter()
  meters = collections.defaultdict(AverageMeter)
  hists = pipeline_model.initialize_hists()

  optimizer = pipeline_model.initialize_optimizer(config)
  scheduler = pipeline_model.initialize_scheduler(optimizer, config)

  writer = SummaryWriter(logdir=config.log_dir)

  # Train the network
  logging.info('===> Start training')
  best_val, best_val_iter, curr_iter, epoch, is_training = 0, 0, 1, 1, True

  if config.resume:
    if osp.isfile(config.resume):
      logging.info("=> loading checkpoint '{}'".format(config.resume))
      state = torch.load(config.resume)
      curr_iter = state['iteration'] + 1
      epoch = state['epoch']
      pipeline_model.load_state_dict(state['state_dict'])
      if config.resume_optimizer:
        curr_iter = state['iteration'] + 1
        scheduler = pipeline_model.initialize_scheduler(optimizer, config, last_step=curr_iter)
        pipeline_model.load_optimizer(optimizer, state['optimizer'])
      if 'best_val' in state:
        best_val = state['best_val']
        best_val_iter = state['best_val_iter']
      logging.info("=> loaded checkpoint '{}' (epoch {})".format(config.resume, state['epoch']))
    else:
      logging.info("=> no checkpoint found at '{}'".format(config.resume))

  data_iter = data_loader.__iter__()
  while is_training:
    for iteration in range(len(data_loader)):
      pipeline_model.reset_gradient(optimizer)
      iter_timer.tic()

      pipelines = parallel.replicate(pipeline_model, devices)

      # Get training data
      data_timer.tic()
      inputs = []
      for pipeline, device in zip(pipelines, devices):
        with torch.cuda.device(device):
          while True:
            datum = pipeline.load_datum(data_iter, has_gt=True)
            num_boxes = sum(box.shape[0] for box in datum['bboxes_coords'])
            if config.skip_empty_boxes and num_boxes == 0:
              continue
            break
          inputs.append(datum)
      data_time_avg.update(data_timer.toc(False))

      outputs = parallel.parallel_apply(pipelines, [(x, True) for x in inputs], devices=devices)
      losses = parallel.parallel_apply([pipeline.loss for pipeline in pipelines],
                                       tuple(zip(inputs, outputs)), devices=devices)
      losses = parallel.gather(losses, target_device)
      losses = dict([(k, v.mean()) for k, v in losses.items()])

      meters, hists = pipeline_model.update_meters(meters, hists, losses)

      # Compute and accumulate gradient
      losses['loss'].backward()

      # Update number of steps
      pipeline_model.step_optimizer(losses, optimizer, scheduler, iteration)

      iter_time_avg.update(iter_timer.toc(False))

      if curr_iter >= config.max_iter:
        is_training = False
        break

      if curr_iter % config.stat_freq == 0 or curr_iter == 1:
        lrs = ', '.join(['{:.3e}'.format(x) for x in scheduler['default'].get_lr()])
        debug_str = "===> Epoch[{}]({}/{}): LR: {}\n".format(epoch, curr_iter, len(data_loader),
                                                             lrs)
        debug_str += log_meters(meters, log_perclass_meters=False)
        debug_str += f"\n    data time: {data_time_avg.avg:.3f}"
        debug_str += f"    iter time: {iter_time_avg.avg:.3f}"
        logging.info(debug_str)

        # Reset timers
        data_time_avg.reset()
        iter_time_avg.reset()

        # Write logs
        update_writer(writer, meters, curr_iter, 'training')
        writer.add_scalar('training/learning_rate', scheduler['default'].get_lr()[0], curr_iter)

        # Reset meters
        reset_meters(meters, hists)

      # Save current status, save before val to prevent occational mem overflow
      if curr_iter % config.save_freq == 0:
        checkpoint(pipeline_model, optimizer, epoch, curr_iter, config, best_val, best_val_iter)

      if config.heldout_save_freq > 0 and curr_iter % config.heldout_save_freq == 0:
        checkpoint(pipeline_model, optimizer, epoch, curr_iter, config, best_val, best_val_iter,
                   heldout_save=True)

      # Validation
      if curr_iter % config.val_freq == 0:
        if num_devices > 1:
          unconvert_sync_batchnorm(pipeline_model)
        best_val, best_val_iter = validate(pipeline_model, val_data_loader, config, writer,
                                           curr_iter, best_val, best_val_iter, optimizer, epoch)
        if num_devices > 1:
          pipeline_model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(pipeline_model, devices)

      if curr_iter % config.empty_cache_freq == 0:
        # Clear cache
        torch.cuda.empty_cache()

      # End of iteration
      curr_iter += 1

    epoch += 1

  # Explicit memory cleanup
  if hasattr(data_iter, 'cleanup'):
    data_iter.cleanup()

  # Save the final model
  if num_devices > 1:
    unconvert_sync_batchnorm(pipeline_model)
  validate(pipeline_model, val_data_loader, config, writer, curr_iter, best_val, best_val_iter,
           optimizer, epoch)
  if num_devices > 1:
    pipeline_model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(pipeline_model, devices)
  checkpoint(pipeline_model, optimizer, epoch, curr_iter, config, best_val, best_val_iter)
