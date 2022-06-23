import os
import pathlib

import numpy as np
import torch
import torch.nn as nn

import lib.utils as utils
import lib.pc_utils as pc_utils
import models

from lib.detection_ap import DetectionAPCalculator
from lib.pipelines.detection import BasePipeline, FasterRCNN
from lib.instance_ap import InstanceAPCalculator


class InstanceSegmentation(BasePipeline):

  TARGET_METRIC = 'mAP@0.50'

  def get_metric(self, val_dict):
    return val_dict['ap_inst'].evaluate()['all_ap_50%']

  def initialize_hists(self):
    return {
        'ap_inst': InstanceAPCalculator(),
        'ap_det': DetectionAPCalculator(0.5),
    }

  def evaluate(self, datum, output):
    instance_target = datum['instance_target'].cpu().numpy()
    gt_instances = [(instance_target == i, gt_class)
                    for i, gt_class in enumerate(datum['bboxes_cls'][0])]
    return {'inst': {'gt': gt_instances, 'pred': output.get('pred_instances', [])},
            'det': super().evaluate(datum, output)['ap']}

  @staticmethod
  def update_meters(meters, hists, loss_dict):
    for k, v in loss_dict.items():
      if k == 'inst':
        hists['ap_inst'].step(v['pred'], v['gt'])
        meters['ap_inst'] = hists['ap_inst']
      elif k == 'det':
        hists['ap_det'].step(v['pred'], v['gt'])
        meters['ap_det'] = hists['ap_det']
      else:
        meters[k].update(v)
    return meters, hists

  def _get_mask_target(self, batch_bboxes, batch_coords, batch_bboxes_id=None,
                       batch_instance_target=None, is_train=True):
    # TODO(jgwak): Incorporate rotation in mask target extraction.
    mask_coords = []
    mask_coordmasks = []
    mask_insts = []
    batch_coords = batch_coords.to(self.device)
    for batch_idx, bboxes in enumerate(batch_bboxes):
      bboxes = torch.from_numpy(bboxes).to(self.device)
      batch_mask = batch_coords[:, 0] == batch_idx
      coords = batch_coords[batch_mask, 1:]
      if batch_bboxes_id is not None and batch_instance_target is not None:
        instance_target = batch_instance_target[batch_mask]
        bboxes_id = batch_bboxes_id[batch_idx]
      batch_mask_coords = []
      batch_mask_coordmasks = []
      batch_mask_insts = []
      coords_aug = coords[..., None].repeat(1, 1, len(bboxes)).permute(0, 2, 1)
      box_masks = (torch.all(coords_aug > bboxes[:, :3], 2)
                   & torch.all(coords_aug < bboxes[:, 3:6], 2)).T
      for bbox_idx, box_mask in enumerate(box_masks):
        batch_mask_coordmasks.append(box_mask)
        batch_mask_coords.append(coords[box_mask])
        if batch_bboxes_id is not None and batch_instance_target is not None:
          batch_mask_insts.append(instance_target[box_mask] == bboxes_id[bbox_idx])
      mask_coords.append(batch_mask_coords)
      mask_coordmasks.append(batch_mask_coordmasks)
      mask_insts.append(batch_mask_insts)
    if is_train and len(mask_coords) <= 1:
      mask_coords, mask_insts, mask_coordmasks = [], [], []
    return mask_coords, mask_insts, mask_coordmasks

  def visualize_groundtruth(self, datum, iteration):
    super().visualize_groundtruth(datum, iteration)
    coords = datum['coords'].numpy()
    batch_size = coords[:, 0].max() + 1
    output_path = pathlib.Path(self.config.visualize_path)
    for i in range(batch_size):
      coords_mask = coords[:, 0] == i
      coords_b = coords[coords_mask, 1:]
      instance_idxs = datum['instance_target'].cpu().numpy()
      instance_mask = instance_idxs != self.config.ignore_label
      if np.any(instance_mask):
        inst_ptc = pc_utils.colorize_pointcloud(
            coords_b[instance_mask], instance_idxs[instance_mask], repeat_color=True)
        inst_dest = output_path / ('visualize_%04d_inst_gt.ply' % iteration)
        pc_utils.save_point_cloud(inst_ptc, inst_dest)

  def visualize_predictions(self, datum, output, iteration):
    super().visualize_predictions(datum, output, iteration)
    batch_size = datum['coords'].numpy()[:, 0].max() + 1
    output_path = pathlib.Path(self.config.visualize_path)
    for i in range(batch_size):
      if output.get('pred_instances') is not None:
        inst_coords, inst_idxs = [], []
        for j, (mask_pred, _, _) in enumerate(output.get('pred_instances')):
          inst_coords.append(datum['coords'][:, 1:].numpy()[mask_pred])
          inst_idxs.append(np.ones(mask_pred.sum(), dtype=int) * j)
        inst_ptc = pc_utils.colorize_pointcloud(np.vstack(inst_coords), np.concatenate(inst_idxs),
                                                repeat_color=True)
        inst_dest = output_path / ('visualize_%04d_inst_pred.ply' % iteration)
        pc_utils.save_point_cloud(inst_ptc, inst_dest)


class MaskRCNN(InstanceSegmentation, FasterRCNN):
  def __init__(self, config, dataset):
    super().__init__(config, dataset)
    self.trilinearalign = models.instance.PyramidTrilinearInterpolation(
        self.feature_upsample_network.out_channels, self.feature_upsample_network.OUT_PIXEL_DIST,
        config).to(self.device)
    self.mask_network = models.instance.MaskNetwork(
        self.trilinearalign.out_channels, config).to(self.device)
    self.mask_criterion = nn.BCEWithLogitsLoss()

  def forward(self, datum, is_train):
    output = super().forward(datum, is_train)
    generate_proposal = not (is_train and self.config.train_rpnonly)
    if generate_proposal:
      if is_train:
        if output.get('ref_mask_coords') is not None:
          ref_mask_sinput = self.trilinearalign(
              output['unnorm_ref_posrois'], output['ref_mask_coords'], output['fpn_outputs'])
          if ref_mask_sinput is not None:
            output['ref_mask_pred'] = self.mask_network(ref_mask_sinput).F.flatten()
      else:
        if sum([detection.shape[0] for detection in output['detection']]):
          detection_mask_sinput = self.trilinearalign(
              output['detection'], output['mask_coords'], output['fpn_outputs'])
          if detection_mask_sinput is not None:
            output['detection_mask_pred'] = self.mask_network(detection_mask_sinput).F.flatten()
            mask_score = torch.sigmoid(output['detection_mask_pred'])
            mask_pred = mask_score > self.config.mask_min_confidence
            mask_coords = output['mask_coords'][0]
            box_mask = torch.cat([torch.ones(mask_coord.shape[0]) * i
                                  for i, mask_coord in enumerate(mask_coords)])
            mask_coords = torch.cat(mask_coords)
            pred_class_scores = output['detection'][0][:, -2:]
            pred_masks = []
            pred_classes = []
            pred_scores = []
            for i in range(int(box_mask.max().item() + 1)):
              coord_masks = output['mask_coordmasks'][0][i].cpu().numpy()
              batch_pred = mask_pred[box_mask == i].cpu().numpy()
              pred_instance = np.full(coord_masks.shape, False, dtype=bool)
              pred_instance[coord_masks] = batch_pred
              pred_masks.append(pred_instance)
              pred_classes.append(int(pred_class_scores[i][0]))
              pred_scores.append(pred_class_scores[i][1])
            if self.config.mask_class_nms:
              accepted = []
              for cls in np.unique(pred_classes):
                cls_idx = np.where(np.array(pred_classes) == cls)[0]
                cls_accepted = utils.mask_nms([pred_masks[i] for i in cls_idx],
                                              self.config.mask_nms_threshold)
                accepted.extend([cls_idx[i] for i in cls_accepted])
            else:
              accepted = utils.mask_nms(pred_masks, self.config.mask_nms_threshold)
            output['pred_instances'] = [
                (pred_masks[i], pred_classes[i], pred_scores[i]) for i in sorted(accepted)]
    return output

  def save_prediction(self, datum, output, save_pred_dir, iteration):
    gt_path = os.path.join(save_pred_dir, 'gt')
    os.makedirs(gt_path, exist_ok=True)
    scene_id = os.path.splitext(self.dataset.data_paths[iteration].split(os.sep)[-1])[0]
    with open(os.path.join(gt_path, scene_id + '.txt'), 'w') as f:
      label = datum['pointcloud'][0][:, 6:]
      f.write('\n'.join(str(int(i)) for i in (label[:, 0] * 1000 + label[:, 1]).numpy()))
    pred_path = os.path.join(save_pred_dir, 'pred')
    os.makedirs(pred_path, exist_ok=True)
    mask_path = os.path.join(pred_path, 'predicted_masks')
    os.makedirs(mask_path, exist_ok=True)

    voxel_coords = datum['coords'][:, 1:].numpy()
    max_voxel_coords = voxel_coords.max(0) + 1
    voxel_coords_idx = np.ravel_multi_index(voxel_coords.T, max_voxel_coords)
    voxel_coords_ndx = voxel_coords_idx.argsort()
    coords = datum['pointcloud'][0][:, :3]
    coords = np.floor(np.hstack((coords, np.ones((coords.shape[0], 1))))
                      @ datum['transformation'][0][0].reshape(4, 4).numpy().T[:, :3])
    coords_idx = np.ravel_multi_index(coords.T.astype(int), max_voxel_coords)
    coords_idx = voxel_coords_ndx[np.searchsorted(voxel_coords_idx[voxel_coords_ndx], coords_idx)]
    mask_strs = []
    if 'pred_instances' in output:
      for i, (coords_mask, mask_class, mask_score) in enumerate(output['pred_instances']):
        mask_f = os.path.join('predicted_masks', scene_id + '_%05d.txt' % i)
        with open(os.path.join(pred_path, mask_f), 'w') as f:
          f.write('\n'.join(coords_mask[coords_idx].astype(int).astype(str)))
        class_s = self.dataset.INSTANCE_LABELS[mask_class]
        mask_strs.append(f'{mask_f} {class_s} {mask_score}')
    with open(os.path.join(pred_path, scene_id + '.txt'), 'w') as f:
      f.write('\n'.join(mask_strs))

  def loss(self, datum, output):
    faster_rcnn_loss = super().loss(datum, output)
    if 'ref_mask_pred' in output and output['ref_mask_pred'] is not None:
      ref_mask_target = torch.cat(
          [item for sublist in output['ref_mask_insts'] for item in sublist])
      mask_loss = self.mask_criterion(output['ref_mask_pred'], ref_mask_target.float())
      faster_rcnn_loss['loss'] += mask_loss * self.config.mask_weight
    else:
      mask_loss = torch.scalar_tensor(0.).to(self.device)
    faster_rcnn_loss['mask_loss'] = mask_loss
    return faster_rcnn_loss


class MaskRCNN_PointNet(MaskRCNN):
  def __init__(self, config, dataset):
    super().__init__(config, dataset)
    self.mask_network = models.pointnet.PointNet(
        self.trilinearalign.out_channels, 1, config).to(self.device)


class MaskRCNN_PointNetXS(MaskRCNN):
  def __init__(self, config, dataset):
    super().__init__(config, dataset)
    self.mask_network = models.pointnet.PointNetXS(
        self.trilinearalign.out_channels, 1, config).to(self.device)
