import errno
import json
import logging
import os
import random
import time
import warnings

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from lib.detection_ap import DetectionAPCalculator
from lib.instance_ap import InstanceAPCalculator
import MinkowskiEngine as ME


def elementwise_multiplication(x, y, n):

  def is_iterable(z):
    if isinstance(z, (list, tuple)):
      return True
    else:
      assert type(z) is int
      return False

  if is_iterable(x) and is_iterable(y):
    assert len(x) == len(y) == n

  def convert_to_iterable(z):
    if is_iterable(z):
      return z
    else:
      return [
          z,
      ] * n

  x = convert_to_iterable(x)
  y = convert_to_iterable(y)
  return [a * b for a, b in zip(x, y)]


def load_state_with_same_shape(model, weights):
  model_state = model.state_dict()
  filtered_weights = {
      k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
  }
  logging.info("Loading weights:" + ', '.join(filtered_weights.keys()))
  return filtered_weights


def unconvert_sync_batchnorm(module):
  module_output = module
  if isinstance(module, ME.MinkowskiSyncBatchNorm):
      module_output = ME.MinkowskiBatchNorm(
          module.bn.num_features, module.bn.eps, module.bn.momentum,
          module.bn.affine, module.bn.track_running_stats)
      if module.bn.affine:
          module_output.bn.weight.data = module.bn.weight.data.clone(
          ).detach()
          module_output.bn.bias.data = module.bn.bias.data.clone().detach(
          )
          # keep reuqires_grad unchanged
          module_output.bn.weight.requires_grad = module.bn.weight.requires_grad
          module_output.bn.bias.requires_grad = module.bn.bias.requires_grad
      module_output.bn.running_mean = module.bn.running_mean
      module_output.bn.running_var = module.bn.running_var
      module_output.bn.num_batches_tracked = module.bn.num_batches_tracked
  for name, child in module.named_children():
      module_output.add_module(name, unconvert_sync_batchnorm(child))
  del module
  return module_output


def checkpoint(model,
               optimizer,
               epoch,
               iteration,
               config,
               best_val=None,
               best_val_iter=None,
               postfix=None,
               heldout_save=False):
  mkdir_p(config.log_dir)
  base_filename = f'checkpoint_{config.pipeline}_{config.backbone_model}'
  if not heldout_save and config.overwrite_weights:
    if postfix is not None:
      filename = f"{base_filename}{postfix}.pth"
    else:
      filename = f"{base_filename}.pth"
  else:
    filename = f"{base_filename}_iter_{iteration}.pth.pth"
  checkpoint_file = config.log_dir + '/' + filename
  state = {
      'iteration': iteration,
      'epoch': epoch,
      'pipeline': config.pipeline,
      'backbone': config.backbone_model,
      'state_dict': model.state_dict(),
      'optimizer': dict([(k, v.state_dict()) for k, v in optimizer.items()]),
  }
  if best_val is not None:
    state['best_val'] = best_val
    state['best_val_iter'] = best_val_iter
  json.dump(vars(config), open(config.log_dir + '/config.json', 'w'), indent=4)
  torch.save(state, checkpoint_file)
  logging.info(f"Checkpoint saved to {checkpoint_file}")
  # Delete symlink if it exists
  if os.path.exists(f'{config.log_dir}/weights.pth'):
    os.remove(f'{config.log_dir}/weights.pth')
  # Create symlink
  os.system(f'cd {config.log_dir}; ln -s {filename} weights.pth')


def update_writer(writer, meters, curr_iter, phase):
  for k, v in meters.items():
    if isinstance(v, np.ndarray):
      v_val = np.nanmean(v)
    elif isinstance(v, AverageMeter):
      v_val = v.avg
    elif isinstance(v, DetectionAPCalculator):
      v_val = v.compute_metrics()['mAP']
    elif isinstance(v, InstanceAPCalculator):
      v_val = v.evaluate()['all_ap_50%']
    else:
      raise ValueError(f'Unknown meter value type: {type(v)}')
    writer.add_scalar(f'{phase}/{k}', v_val, curr_iter)


def reset_meters(meters, hists):
  for k, v in meters.items():
    if isinstance(v, np.ndarray):
      meters[k] = np.zeros_like(v)
    elif isinstance(v, AverageMeter):
      v.reset()
    else:
      raise ValueError(f'Unknown meter value type: {type(v)}')
  for k, v in hists.items():
    if isinstance(v, np.ndarray):
      hists[k] = np.zeros_like(v)
    elif isinstance(v, DetectionAPCalculator):
      hists[k].reset()
    elif isinstance(v, InstanceAPCalculator):
      hists[k].reset()
    else:
      raise ValueError(f'Unknown hist value type: {type(v)}')


def log_meters(meters, log_perclass_meters=False, line_limit=90):
  log_str = ''
  for k, v in meters.items():
    if isinstance(v, np.ndarray):
      if not log_perclass_meters:
        new_log = f"    {k}: {np.nanmean(v):.3f}"
      else:
        new_log = ''
    elif isinstance(v, AverageMeter):
      new_log = f"    {k}: {v.avg:.3f}"
    elif isinstance(v, DetectionAPCalculator):
      det_stats = v.compute_metrics()
      class_idx = 0
      class_aps = []
      class_recalls = []
      while True:
        class_ap = det_stats.get(f'{class_idx} Average Precision')
        class_recall = det_stats.get(f'{class_idx} Recall')
        if class_ap is None:
          break
        class_aps.append(class_ap)
        class_recalls.append(class_recall)
        class_idx += 1
      new_log = f"    {k} mAP: {det_stats['mAP']:.04f} AR: {det_stats['AR']:.04f}"
      new_log += f'\n          AP:  ' + ' '.join('{:.04f}'.format(i) for i in class_aps)
      new_log += f'\n          R :  ' + ' '.join('{:.04f}'.format(i) for i in class_recalls)
    elif isinstance(v, InstanceAPCalculator):
      new_log = f"    {k}: {v.evaluate()['all_ap_50%']:.3f}"
    else:
      raise ValueError(f'Unknown meter value type: {type(v)}')
    if len(log_str.split('\n')[-1]) + len(new_log) > line_limit:
      log_str += '\n'
    log_str += new_log
  if log_perclass_meters:
    for k, v in meters.items():
      if isinstance(v, np.ndarray):
        log_str += f"\n    mean {k}: {np.nanmean(v):.3f}"
        log_str += '\n        ' + ' '.join('{:.03f}'.format(i) for i in v)
  return log_str


def feat_augmentation(data, normalized, config):
  # color shift
  if random.random() < 0.9:
    tr = (torch.rand(1, 3).type_as(data) - 0.5) * \
        config.data_aug_max_color_trans
    if normalized:
      tr /= 255
    data[:, :3] += tr

  # color jitter
  if random.random() < 0.9:
    noise = torch.randn((data.size(0), 3), dtype=data.dtype).type_as(data)
    noise *= config.data_aug_noise_std if normalized else 255 * config.data_aug_noise_std
    data[:, :3] += noise

  # height jitter
  if data.size(1) > 3 and random.random() < 0.9:
    data[:, -1] += torch.randn(1, dtype=data.dtype).type_as(data)

  if data.size(1) > 3 and random.random() < 0.9:
    data[:, -1] += torch.randn((data.size(0)), dtype=data.dtype).type_as(data)

  # normal jitter
  if data.size(1) > 6 and random.random() < 0.9:
    data[:, 3:6] += torch.randn((data.size(0), 3), dtype=data.dtype).type_as(data)


def precision_at_one(pred, target, ignore_label=-1):
  """Computes the precision@k for the specified values of k"""
  # batch_size = target.size(0) * target.size(1) * target.size(2)
  pred = pred.view(1, -1)
  target = target.view(1, -1)
  correct = pred.eq(target)
  correct = correct[target != ignore_label]
  correct = correct.view(-1)
  if correct.nelement():
    return correct.float().sum(0).mul(100.0 / correct.size(0)).item()
  else:
    return float('nan')


def fast_hist(pred, label, n):
  k = (label >= 0) & (label < n)
  return np.bincount(n * label[k].astype(int) + pred[k], minlength=n**2).reshape(n, n)


def per_class_iu(hist):
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def get_panoptic_quality(panoptic_hist):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    iou_tp = panoptic_hist[:, 0]
    tp = panoptic_hist[:, 1]
    fp = panoptic_hist[:, 2]
    fn = panoptic_hist[:, 3]
    sq = iou_tp / tp
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    pq = iou_tp / (tp + 0.5 * fp + 0.5 * fn)
    return pq, sq, rq


def get_panoptic_hist(panoptic_pred, bboxes_pred, semantic_target, instance_target, dataset,
                      ignore_label):
  num_stuff_class = dataset.NUM_LABELS - len(dataset.INSTANCE_LABELS)
  num_thing_pred = bboxes_pred.shape[0]
  panoptic_pred[semantic_target == ignore_label] = ignore_label
  # For each label, accumulate TP IoU, TP, FP, FN.
  panoptic_hist = np.zeros((dataset.NUM_LABELS, 4))
  # Evaluate stuff class.
  for stuff_idx in range(num_stuff_class):
    pred_mask = panoptic_pred == stuff_idx
    gt_mask = semantic_target == stuff_idx
    union = (gt_mask | pred_mask).sum().item()
    # No prediction nor ground-truth label. True negative. Ignored in panoptic metrics.
    if union == 0:
      continue
    iou = (gt_mask & pred_mask).sum().item() / union
    # Check if matched.
    if iou > 0.5:
      panoptic_hist[stuff_idx, 0] = iou
      panoptic_hist[stuff_idx, 1] = 1
    else:
      # If there is prediction, it is false positive.
      if pred_mask.sum().item() > 0:
        panoptic_hist[stuff_idx, 2] = 1
      # If there is ground-truth label, it is false negative.
      if gt_mask.sum().item() > 0:
        panoptic_hist[stuff_idx, 3] = 1
  # Evaluate things class.
  for thing_class_idx in range(num_stuff_class, dataset.NUM_LABELS):
    # Look for panoptic things prediction index for the current semantic class.
    things_class_mask = (bboxes_pred[:, 6] == thing_class_idx).numpy().astype(bool)
    things_pred_idx = np.arange(num_thing_pred)[things_class_mask]
    # Look for instance label index for the current semantic class.
    gt_semantic_mask = semantic_target == thing_class_idx
    if gt_semantic_mask.sum():
      things_gt_idx = torch.unique(instance_target[gt_semantic_mask])
    else:
      things_gt_idx = torch.zeros(0)
    # Try to match prediction to ground-truth instance.
    things_pred_match = np.zeros(things_pred_idx.size)
    things_gt_match = np.zeros(things_gt_idx.size())
    for thing_pred_idx, thing_panoptic_idx in enumerate(things_pred_idx):
      for thing_gt_idx, thing_instance_idx in enumerate(things_gt_idx):
        # Ignore if there is a match.
        if things_gt_match[thing_gt_idx]:
          continue
        # Compute IoU between the prediction and ground-truth.
        gt_mask = instance_target == thing_instance_idx
        pred_mask = panoptic_pred == thing_panoptic_idx + num_stuff_class
        union = (gt_mask | pred_mask).sum().item()
        assert union > 0  # There MUST be something.
        iou = (gt_mask & pred_mask).sum().item() / union
        if iou > 0.5:  # If matched, update the corresponding statistics.
          panoptic_hist[thing_class_idx, 0] += iou
          panoptic_hist[thing_class_idx, 1] += 1
          things_pred_match[thing_pred_idx] = 1
          things_gt_match[thing_gt_idx] = 1
    # Any predictions with no match is false positive.
    panoptic_hist[thing_class_idx, 2] = (things_pred_match == 0).sum()
    # Any ground-truth with no match is false negative.
    panoptic_hist[thing_class_idx, 3] = (things_gt_match == 0).sum()
  return panoptic_hist


class WithTimer(object):
  """Timer for with statement."""

  def __init__(self, name=None):
    self.name = name

  def __enter__(self):
    self.tstart = time.time()

  def __exit__(self, type, value, traceback):
    out_str = 'Elapsed: %s' % (time.time() - self.tstart)
    if self.name:
      logging.info('[{self.name}]')
    logging.info(out_str)


class Timer(object):
  """A simple timer."""

  def __init__(self):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.average_time = 0.

  def reset(self):
    self.total_time = 0
    self.calls = 0
    self.start_time = 0
    self.diff = 0
    self.averate_time = 0

  def tic(self):
    # using time.time instead of time.clock because time time.clock
    # does not normalize for multithreading
    self.start_time = time.time()

  def toc(self, average=True):
    self.diff = time.time() - self.start_time
    self.total_time += self.diff
    self.calls += 1
    self.average_time = self.total_time / self.calls
    if average:
      return self.average_time
    else:
      return self.diff


class ExpTimer(Timer):
  """ Exponential Moving Average Timer """

  def __init__(self, alpha=0.5):
    super(ExpTimer, self).__init__()
    self.alpha = alpha

  def toc(self):
    self.diff = time.time() - self.start_time
    self.average_time = self.alpha * self.diff + \
        (1 - self.alpha) * self.average_time
    return self.average_time


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    if n > 0:
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count


def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


def debug_on():
  import sys
  import pdb
  import functools
  import traceback

  def decorator(f):

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      try:
        return f(*args, **kwargs)
      except Exception:
        info = sys.exc_info()
        traceback.print_exception(*info)
        pdb.post_mortem(info[2])

    return wrapper

  return decorator


def average_precision(prob_np, label, binarize=True):
  if binarize:
    num_class = prob_np.shape[1]
    label = label_binarize(label, classes=list(range(num_class)))
  with np.errstate(divide='ignore', invalid='ignore'):
    return average_precision_score(label, prob_np, None) * 100.


def compute_iou_3d(box1, box2):
  """ Compute 3D IoU.
  Args:
    box1, box2: (6,) shape array storing (centerX, centerY, centerZ, w, h, l)
  Returns:
    3D IOU
  """
  def get_overlap(left1, right1, left2, right2):
    left_max = max(left1, left2)
    right_min = min(right1, right2)
    overlap = max(right_min - left_max, 0)
    return overlap

  x1, y1, z1, w1, h1, l1 = box1
  x2, y2, z2, w2, h2, l2 = box2
  vol1 = w1 * h1 * l1
  vol2 = w2 * h2 * l2
  x_inter = get_overlap(x1 - w1 / 2, x1 + w1 / 2, x2 - w2 / 2, x2 + w2 / 2)
  y_inter = get_overlap(y1 - h1 / 2, y1 + h1 / 2, y2 - h2 / 2, y2 + h2 / 2)
  z_inter = get_overlap(z1 - l1 / 2, z1 + l1 / 2, z2 - l2 / 2, z2 + l2 / 2)
  inter_vol = x_inter * y_inter * z_inter
  union_vol = vol1 + vol2 - inter_vol
  return inter_vol / union_vol


def get_submask(mask1, mask2):
  mask = torch.zeros_like(mask1).to(mask2)
  mask[mask1] = mask2
  return mask


def match_bboxes(bboxes_pred, bboxes_gt, iou_thresh):
  matched_bboxes_pred = []
  for bbox_pred in bboxes_pred:
    for i, bbox_gt in enumerate(bboxes_gt[bboxes_gt[:, -1] == bbox_pred[-1]]):
      if bbox_gt[6] != bbox_pred[6]:  # Check if semantic classes match.
        continue
      # Check if ground-truth the bounding box overlaps.
      if compute_iou_3d(bbox_pred[:6], bbox_gt[:6]) > iou_thresh:
        matched_bboxes_pred.append(np.insert(bbox_pred, -1, i))
  return np.array(matched_bboxes_pred)


def filter_mask(coords, feats, bboxes, inst_target=None, gt_bbox=False):
  """Generate input of mask branch by filtering coordinates and features based on bounding boxes."""
  coords_mask = []
  feats_mask = []
  target_mask = []
  bboxes_mask = []
  for i, bbox in enumerate(bboxes):
    # Filter coordinates that belongs to the same batch as the bounding box.
    coords_batch_mask = coords[:, -1] == bbox[-1].type(coords.dtype)
    coords_batch = coords[coords_batch_mask, :-1].type(bbox.dtype)
    # Filter coordinates within the given bounding box.
    bbox_mask = torch.all(torch.cat((bbox[:3] - bbox[3:6] < coords_batch,
                                     coords_batch < bbox[:3] + bbox[3:6]), 1), 1)
    bbox_batch_mask = get_submask(coords_batch_mask, bbox_mask)
    bboxes_mask.append(bbox_batch_mask)
    # Put all filtered coordinates and features together.
    coords_mask.append(
        torch.cat((coords[bbox_batch_mask, :-1],
                   torch.ones((bbox_batch_mask.sum(), 1)).type(coords.dtype) * i), 1))
    feats_mask.append(feats.F[bbox_batch_mask])
    if inst_target is not None:
      inst_bbox = inst_target[bbox_batch_mask.to(inst_target).type(coords_batch_mask.dtype)]
      gt_idx = i - (bboxes[:, -1] < bbox[-1]).sum().item()
      gt_bbox_idx = gt_idx if gt_bbox else bbox[-2].to(inst_bbox)
      target_mask.append(inst_bbox == gt_bbox_idx)
  if coords_mask:
    coords_mask = torch.cat(coords_mask, 0).type(coords.dtype)
    feats_mask = torch.cat(feats_mask, 0)
    bboxes_mask = torch.stack(bboxes_mask)
  else:
    coords_mask = torch.zeros((0, coords.shape[1]))
    feats_mask = torch.zeros((0, feats.F.shape[1]))
    bboxes_mask = torch.zeros(0)
  return_args = (coords_mask, feats_mask, bboxes_mask)
  if inst_target is not None:
    return_args += (torch.cat(target_mask, 0) if target_mask else torch.zeros(0),)
  return return_args


def get_panoptic_head(semantic_output, bboxes, bbox_filters, mask_coords,
                      mask_output, dataset, ignore_label=-1, unknown_ratio=0., semantic_target=None,
                      mask_target=None):
  compute_target = semantic_target is not None and mask_target is not None
  num_stuff_class = dataset.NUM_LABELS - len(dataset.INSTANCE_LABELS)
  num_things = bboxes.shape[0]
  num_panoptic_class = num_stuff_class + num_things
  num_points = semantic_output.shape[0]
  unknown_idx = num_panoptic_class
  # Initialize return values.
  num_panoptic_logits = num_panoptic_class if unknown_ratio <= 0 else num_panoptic_class + 1
  panoptic_logits = torch.zeros(num_points, num_panoptic_logits).to(semantic_output)
  semantic_masks = torch.zeros(num_points, num_things).to(semantic_output)
  if compute_target:
    panoptic_gt = torch.ones(num_points).to(semantic_output).long() * ignore_label
  # Copy semantic logits and label from stuff class.
  panoptic_logits[:, :num_stuff_class] = semantic_output[:, :num_stuff_class]
  if compute_target:
    stuff_mask = semantic_target < num_stuff_class
    panoptic_gt[stuff_mask] = semantic_target[stuff_mask]
  for i, (bbox, bbox_filter) in enumerate(zip(bboxes, bbox_filters)):
    mask_filter = mask_coords[:, -1] == i
    thing_idx = num_stuff_class + i
    semantic_mask = semantic_output[bbox_filter, bbox[6].long()]
    # Panoptic logit of instance = mask prediction logit + semantic prediction logit.
    panoptic_logits[bbox_filter, thing_idx] = semantic_mask + mask_output[mask_filter]
    # Accumulate masked semantic predictions for unknown logit.
    semantic_masks[bbox_filter, i] = semantic_mask
    # Assign corresponding target label to panoptic things labels, with unknown label perturbation.
    if compute_target:
      instance_filter = get_submask(bbox_filter, mask_target[mask_filter])
      panoptic_gt[instance_filter] = unknown_idx if np.random.rand() < unknown_ratio else thing_idx
  # Unknown logit = max(semantic logits of things) - max(masked semantic predictions).
  if semantic_output.shape[0] and unknown_ratio > 0:
    panoptic_logits[:, -1] = semantic_output[:, num_stuff_class:].max(1)[0]
    if num_things:
      panoptic_logits[:, -1] -= semantic_masks.max(1)[0]
  if compute_target:
    return panoptic_logits, panoptic_gt
  return panoptic_logits


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_torch_device(is_cuda):
  return torch.device('cuda' if is_cuda else 'cpu')


class HashTimeBatch(object):

  def __init__(self, prime=5279):
    self.prime = prime

  def __call__(self, time, batch):
    return self.hash(time, batch)

  def hash(self, time, batch):
    return self.prime * batch + time

  def dehash(self, key):
    time = key % self.prime
    batch = key / self.prime
    return time, batch


def evaluate_temporal_average(output, coords):
  """Take average of output across temporal dimension for the same 3D coordinates."""
  # Take average independently for each batch.
  for i in range(coords[:, -1].max().item() + 1):
    # Mask batch data.
    batch_mask = coords[:, -1] == i
    batch_coords = coords[batch_mask, :3].numpy()
    batch_temporal = coords[batch_mask, -2].numpy()
    assert batch_coords.min() >= 0
    # Get unique coordinates across 3D coordinates.
    ravel_idx = np.ravel_multi_index(batch_coords.T, batch_coords.max(0) + 1)
    _, temporal_idx = np.unique(ravel_idx, return_inverse=True)
    # Unravel output features across temporal dimension.
    num_class = output.shape[1]
    output_temporal = np.empty((temporal_idx.max() + 1, batch_temporal.max() + 1, num_class))
    output_temporal[:] = np.nan
    temporal_ravel_idx = np.ravel_multi_index(
        np.vstack((temporal_idx, batch_temporal)), output_temporal.shape[:2])
    output_temporal.reshape(-1, num_class)[temporal_ravel_idx] = output[batch_mask].cpu().numpy()
    # Take average across the temporal dimension.
    output_temporal = np.nanmean(output_temporal, 1)
    output_temporal = np.take(output_temporal, temporal_idx, axis=0)
    assert not np.any(np.isnan(output_temporal))
    # Assign averaged values back to the output.
    output[batch_mask] = torch.from_numpy(output_temporal).to(output)


def mask_nms(pred_masks, mask_nms_threshold):
  num_masks = len(pred_masks)
  if num_masks == 0:
    return []
  mask_diag = torch.from_numpy(np.stack(pred_masks)[
      np.transpose(np.triu_indices(num_masks, 1))]).permute(1, 0, 2).cuda()
  mask_intersection = (mask_diag[0] & mask_diag[1]).sum(1).float()
  mask_union = (mask_diag[0] | mask_diag[1]).sum(1).float()
  mask_overlap = (mask_intersection / mask_union) > mask_nms_threshold
  overlap = np.full((num_masks, num_masks), False, dtype=bool)
  overlap[np.triu_indices(num_masks, 1)] = mask_overlap.cpu().numpy()
  accepted = [0]
  for i in range(1, num_masks):
    if not np.any(overlap[accepted, i]):
      accepted.append(i)
  return accepted
