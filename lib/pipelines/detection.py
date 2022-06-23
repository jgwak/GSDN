import glob
import itertools
import logging
import os
import pathlib
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lib.detection_utils as detection_utils
import lib.pc_utils as pc_utils
import lib.solvers as solvers
import lib.utils as utils
import models
from lib.detection_ap import DetectionAPCalculator
from lib.pipelines.base import BasePipeline
from lib.loss import get_rotation_loss, get_classification_loss


class Detection(BasePipeline):

  TARGET_METRIC = 'mAP@0.25'
  EVAL_PERCLASS_MAP = True

  def __init__(self, config, dataset):
    super().__init__(config, dataset)

  def get_metric(self, val_dict):
    return val_dict['ap_25'].compute_metrics()['mAP']

  def initialize_hists(self):
    return {
        'ap_25': DetectionAPCalculator(0.25),
        'ap_50': DetectionAPCalculator(0.5),
    }

  def evaluate(self, datum, output):
    if not self.EVAL_PERCLASS_MAP:
      warnings.warn('Evaluating single-class AP. This is NOT a metric comparable to other methods.')
    if self.dataset.IS_ROTATION_BBOX:
      gt_boxes = [[(bbox_cls if self.EVAL_PERCLASS_MAP else 0, bbox) for bbox, bbox_cls in
                   zip(pc_utils.bboxes2corners(np.hstack((bboxes, bboxes_rot[:, None])),
                                               bbox_param='xyzxyzr', swap_yz=True), bboxes_cls)]
                  if bboxes.size != 0 else [] for bboxes, bboxes_rot, bboxes_cls in
                  zip(datum['bboxes_coords'], datum['bboxes_rotations'], datum['bboxes_cls'])]
      pred_boxes = [[(bbox_cls if self.EVAL_PERCLASS_MAP else 0, bbox, bbox_score)
                     for bbox, bbox_cls, bbox_score in
                     zip(pc_utils.bboxes2corners(bboxes[:, :7], bbox_param='xyzxyzr', swap_yz=True),
                         bboxes[:, 7].astype(int), bboxes[:, 8])]
                    if bboxes.size != 0 else [] for bboxes in output['detection']]
    else:
      gt_boxes = [[(bbox_cls if self.EVAL_PERCLASS_MAP else 0, bbox)
                   for bbox, bbox_cls in zip(pc_utils.bboxes2corners(bboxes, swap_yz=True),
                                             bboxes_cls)]
                  if bboxes.size != 0 else []
                  for bboxes, bboxes_cls in zip(datum['bboxes_coords'], datum['bboxes_cls'])]
      pred_boxes = [bboxes[np.all(np.isfinite(bboxes), 1)] for bboxes in output['detection']]
      pred_boxes = [bboxes[:, :self.config.detection_max_instances] for bboxes in pred_boxes]
      pred_boxes = [[(bbox_cls if self.EVAL_PERCLASS_MAP else 0, bbox, bbox_score)
                     for bbox, bbox_cls, bbox_score
                     in zip(pc_utils.bboxes2corners(bboxes[:, :6], swap_yz=True),
                            bboxes[:, 6].astype(int), bboxes[:, 7])]
                    if bboxes.size != 0 else [] for bboxes in pred_boxes]
    return {'ap': {'gt': gt_boxes, 'pred': pred_boxes}}

  def test_original_pointcloud(self, save_pred_dir):
    # TODO: Refactor this method to reuse evaluation code.
    ap_25 = DetectionAPCalculator(0.25)
    ap_50 = DetectionAPCalculator(0.5)

    for fn in glob.glob(os.path.join(save_pred_dir, '*.npz')):
      datum = np.load(fn)
      if datum['pred'].size == 0:
        pred_boxes = [[]]
      else:
        if self.dataset.IS_ROTATION_BBOX:
          pred_boxes = [[(bbox_cls, bbox, bbox_score)
                         for bbox, bbox_cls, bbox_score
                         in zip(pc_utils.bboxes2corners(datum['pred'][:, :7], bbox_param='xyzxyzr',
                                swap_yz=True),
                                datum['pred'][:, 7].astype(int), datum['pred'][:, 8])]]
        else:
          pred_boxes = [[(bbox_cls, bbox, bbox_score)
                         for bbox, bbox_cls, bbox_score
                         in zip(pc_utils.bboxes2corners(datum['pred'][:, :6], swap_yz=True),
                                datum['pred'][:, 6].astype(int), datum['pred'][:, 7])]]
      if datum['gt'].size == 0:
        gt_boxes = [[]]
      else:
        if self.dataset.IS_ROTATION_BBOX:
          gt_boxes = [[(bbox_cls, bbox) for bbox, bbox_cls
                       in zip(pc_utils.bboxes2corners(datum['gt'][:, :7], bbox_param='xyzxyzr',
                              swap_yz=True), datum['gt'][:, 7])]]
        else:
          gt_boxes = [[(bbox_cls, bbox) for bbox, bbox_cls
                       in zip(pc_utils.bboxes2corners(datum['gt'][:, :6], swap_yz=True),
                              datum['gt'][:, 6])]]
      ap_25.step(pred_boxes, gt_boxes)
      ap_50.step(pred_boxes, gt_boxes)
    eval_str = '===> Original point cloud evaluation'
    eval_str += utils.log_meters({'ap_25': ap_25, 'ap_50': ap_50})
    if self.config.save_ap_log:
      ap_25_stat = ap_25.compute_metrics()
      ap_50_stat = ap_50.compute_metrics()
      ap25_detail = []
      ap50_detail = []
      for i in range(self.dataset.NUM_LABELS):
        ap25_detail.append((ap_25_stat['rec'][i], ap_25_stat['prec'][i]))
        ap50_detail.append((ap_50_stat['rec'][i], ap_50_stat['prec'][i]))
      np.savez('ap_details', ap25=ap25_detail, ap50=ap50_detail)
    logging.info(eval_str)

  def visualize_groundtruth(self, datum, iteration):
    coords = datum['coords'].numpy()
    batch_size = coords[:, 0].max() + 1
    output_path = pathlib.Path(self.config.visualize_path)
    output_path.mkdir(exist_ok=True)
    for i in range(batch_size):
      # Visualize ground-truth positive anchors.
      anchors_gt = datum['anchors'][torch.where(datum['rpn_match'].cpu() == 1)[1]]
      anchors_gt_ptc = pc_utils.visualize_bboxes(anchors_gt)
      anchors_gt_ply_dest = output_path / ('visualize_%04d_anchors_gt.ply' % iteration)
      pc_utils.save_point_cloud(anchors_gt_ptc, anchors_gt_ply_dest)
      # Visualize center location of all ground-truth anchors.
      anchors_all = np.unique((datum['anchors'][:, 3:] + datum['anchors'][:, :3]) / 2, axis=0)
      anchors_all_ply_dest = output_path / ('visualize_%04d_all_anchors_centers.ply' % iteration)
      pc_utils.save_point_cloud(anchors_all, anchors_all_ply_dest)
      # Visualize ground-truth positive anchors.
      if datum.get('bboxes_rotations') is None:
        bboxes_gt = pc_utils.visualize_bboxes(datum['bboxes_coords'][i], datum['bboxes_cls'][i])
      else:
        bboxes_gt = np.hstack((datum['bboxes_coords'][i], datum['bboxes_rotations'][i][:, None]))
        bboxes_gt = pc_utils.visualize_bboxes(bboxes_gt, datum['bboxes_cls'][i],
                                              bbox_param='xyzxyzr')
      bboxes_gt_ply_dest = output_path / ('visualize_%04d_bboxes_gt.ply' % iteration)
      pc_utils.save_point_cloud(bboxes_gt, bboxes_gt_ply_dest)
      # Visualize reconstructed ground-truth rpn targets.
      rpn_bbox_anchors = datum['anchors'][(datum['rpn_match'].flatten() == 1).cpu().numpy()]
      rpn_bbox_anchors = detection_utils.normalize_boxes(rpn_bbox_anchors, self.config.max_ptc_size)
      rpn_bbox_target = datum['rpn_bbox'].reshape(-1, 6)
      rpn_bbox_mask = ~torch.all(rpn_bbox_target == 0, 1)
      rpn_bbox_target = rpn_bbox_target[rpn_bbox_mask].cpu().numpy()
      rpn_bbox_target *= np.reshape(self.config.rpn_bbox_std, (1, len(self.config.rpn_bbox_std)))
      rpn_bbox_target = detection_utils.apply_box_deltas(torch.from_numpy(rpn_bbox_anchors),
                                                         torch.from_numpy(rpn_bbox_target),
                                                         self.config.normalize_bbox)
      rpn_bbox_target = detection_utils.unnormalize_boxes(rpn_bbox_target.numpy(),
                                                          self.config.max_ptc_size)
      if datum.get('rpn_rotation') is None:
        bboxes_gt_recon = pc_utils.visualize_bboxes(rpn_bbox_target)
      else:
        rpn_rot_target = datum['rpn_rotation'][i][rpn_bbox_mask].cpu().numpy()
        bboxes_gt_recon = np.hstack((rpn_bbox_target, rpn_rot_target[:, None]))
        bboxes_gt_recon = pc_utils.visualize_bboxes(bboxes_gt_recon, bbox_param='xyzxyzr')
      bboxes_gt_recon_ply_dest = output_path / ('visualize_%04d_bboxes_gt_recon.ply' % iteration)
      pc_utils.save_point_cloud(bboxes_gt_recon, bboxes_gt_recon_ply_dest)

  def visualize_predictions(self, datum, output, iteration):
    coords = datum['coords'].numpy()
    batch_size = coords[:, 0].max() + 1
    output_path = pathlib.Path(self.config.visualize_path)
    output_path.mkdir(exist_ok=True)
    for i in range(batch_size):
      # Visualize RGB input.
      coords_mask = coords[:, 0] == i
      coords_b = coords[coords_mask, 1:]
      if datum['input'].shape[1] > 3:
        rgb_b = ((datum['input'][:, :3] + 0.5) * 255).numpy()
      else:
        rgb_b = ((datum['input'].repeat(1, 3) - datum['input'].min())
                 / (datum['input'].max() - datum['input'].min()) * 255).numpy()
      rgb_ply_dest = output_path / ('visualize_%04d_rgb.ply' % iteration)
      pc_utils.save_point_cloud(np.hstack((coords_b, rgb_b)), rgb_ply_dest)
      # Visualize positive anchor centers.
      positive_anchors_mask = F.softmax(output['rpn_class_logits'].cpu(), dim=2)[0, :, 1] > 0.5
      positive_anchors = datum['anchors'][torch.where(positive_anchors_mask)]
      if len(positive_anchors.shape) == 1:
        positive_anchors = np.array([positive_anchors])
      if positive_anchors.shape[0] > 0:
        positive_anchors = np.unique((positive_anchors[:, 3:] + positive_anchors[:, :3]) / 2,
                                     axis=0)
        anchors_all_ply_dest = output_path / ('visualize_%04d_anchors_pred.ply' % iteration)
        pc_utils.save_point_cloud(positive_anchors, anchors_all_ply_dest)
      # Visualize region proposals.
      rpn_rois = output['rpn_rois'][i].cpu().numpy()
      rpn_mask = ~np.all(rpn_rois == 0, 1)
      rpn_rois = rpn_rois[rpn_mask][:100]
      rpn_rois = detection_utils.unnormalize_boxes(rpn_rois, self.config.max_ptc_size)
      if output.get('rpn_rois_rotation') is None:
        rpn_rois_ptc = pc_utils.visualize_bboxes(rpn_rois)
      else:
        rpn_rois_rotation = output['rpn_rois_rotation'][i].cpu().numpy()[rpn_mask][:100]
        rpn_rois = np.hstack((rpn_rois, rpn_rois_rotation[:, None]))
        rpn_rois_ptc = pc_utils.visualize_bboxes(rpn_rois, bbox_param='xyzxyzr')
      rpn_rois_ply_dest = output_path / ('visualize_%04d_rpn_rois_pred.ply' % iteration)
      pc_utils.save_point_cloud(rpn_rois_ptc, rpn_rois_ply_dest)
      # Visualize final detection.
      detection_pred = output['detection'][i]
      if detection_pred.shape[1] == 8:
        detection_ptc = pc_utils.visualize_bboxes(detection_pred[:, :6], detection_pred[:, 6])
      elif detection_pred.shape[1] == 9:
        detection_ptc = pc_utils.visualize_bboxes(detection_pred[:, :7], detection_pred[:, 7],
                                                  bbox_param='xyzxyzr')
      else:
        raise ValueError('Unknown bounding box output.')
      detection_dest = output_path / ('visualize_%04d_detection_pred.ply' % iteration)
      pc_utils.save_point_cloud(detection_ptc, detection_dest)

  def save_prediction(self, datum, output, output_dir, iteration):
    # TODO(jgwak): Incorporate rotated bounding box.
    def _unvoxelize_bbox(bbox, transformation, has_gt_bbox):
      bbox_xyz = bbox[:, :6].copy()
      if not has_gt_bbox:
        bbox_xyz += 0.5
        bbox_xyz[:, :3] += 0.5
        bbox_xyz[:, 3:] -= 0.5
      bbox_xyz[:, 3:] = np.maximum(bbox_xyz[:, 3:], bbox_xyz[:, :3] + 0.1)
      bbox_xyz = bbox_xyz.reshape(-1, 3)
      bbox_xyz1 = np.hstack((bbox_xyz, np.ones((bbox_xyz.shape[0], 1))))
      bbox_xyz = np.linalg.solve(transformation, bbox_xyz1.T).T[:, :3].reshape(-1, 6)
      return np.hstack((bbox_xyz, bbox[:, 6:]))
    transformation = datum['transformation'][0][0].reshape(4, 4).numpy()
    pred = output['detection'][0]
    pred_mask = pred[:, -1] > self.config.save_min_confidence
    pred = _unvoxelize_bbox(pred[pred_mask], transformation, self.dataset.HAS_GT_BBOX)
    if self.dataset.HAS_GT_BBOX:
      gt_box_params = []
      gt_boxes = datum['bboxes_coords'][0].copy()
      gt_boxes = gt_boxes.reshape(-1, 3)
      gt_boxes1 = np.hstack((gt_boxes, np.ones((gt_boxes.shape[0], 1))))
      gt_boxes = np.linalg.solve(transformation.reshape(4, 4), gt_boxes1.T).T[:, :3].reshape(-1, 6)
      gt_box_params.append(gt_boxes)
      if self.dataset.IS_ROTATION_BBOX:
        gt_box_params.append(datum['bboxes_rotations'][0][:, None])
      gt_classes = datum['bboxes_cls'][0]
      if self.dataset.IGNORE_LABELS is not None:
        gt_classes = np.array([self.label_map[int(x)] for x in gt_classes], dtype=np.int)
      gt_classes = gt_classes[:, None]
      gt_box_params.append(gt_classes)
      gt = np.hstack(gt_box_params)
    else:
      pointcloud = datum['pointcloud'][0].numpy()
      semantic_labels, instance_labels = pointcloud[:, -2:].T
      if self.dataset.IGNORE_LABELS is not None:
        semantic_labels = np.array([self.dataset.label_map[x] for x in semantic_labels], dtype=np.int)
      instance_mask = self.dataset.get_instance_mask(semantic_labels, instance_labels)
      gt, _ = pc_utils.get_bbox(pointcloud[:, :3], semantic_labels, instance_labels,
                                instance_mask, self.config.ignore_label, is_voxel=False)
    np.savez(output_dir + f'/out_{iteration:03}.npy', pred=pred, gt=gt)

  def load_pretrained_weights(self, state_dict):
    new_state_dict = {}
    backbone_dict = self.backbone.state_dict()
    for k, v in state_dict.items():
      if not k.startswith('backbone.'):
        continue
      k = k[len('backbone.'):]
      if k in backbone_dict:
        new_state_dict[k] = v
    params_unfilled = ', '.join(set(backbone_dict) - set(new_state_dict))
    if params_unfilled:
      logging.info('Backbone network unfilled parameters: ' + params_unfilled)
    backbone_dict.update(new_state_dict)
    self.backbone.load_state_dict(backbone_dict)


class FasterRCNNBase(Detection):

  REGION_PROPOSAL_NETWORK_MODEL = None
  FEATURE_UPSAMPLE_NETWORK_MODEL = None
  REGION_REFINEMENT_NETWORK_MODEL = None

  def __init__(self, config, dataset):
    super().__init__(config, dataset)
    self.train_rpnonly = self.config.train_rpnonly or self.REGION_REFINEMENT_NETWORK_MODEL is None
    self.EVAL_PERCLASS_MAP = not self.train_rpnonly
    self.initialize_losses(config)
    self.initialize_models(config, dataset)

  def initialize_losses(self, config):
    self.rpn_class_criterion = get_classification_loss(config.rpn_classification_loss)(
        ignore_index=config.ignore_label)
    self.rpn_bbox_criterion = nn.SmoothL1Loss()
    min_angle = -np.pi / 4 if config.normalize_rotation else -np.pi
    max_angle = np.pi / 4 if config.normalize_rotation else np.pi
    self.rpn_rotation_criterion = get_rotation_loss(config.rpn_rotation_loss)(
        config.num_rotation_bins, activation='tanh', min_angle=min_angle, max_angle=max_angle)
    self.ref_class_criterion = get_classification_loss(config.ref_classification_loss)(
        ignore_index=config.ignore_label)
    self.ref_bbox_criterion = nn.SmoothL1Loss()
    min_angle = -np.pi / 2 if config.normalize_rotation else -np.pi
    max_angle = np.pi / 2 if config.normalize_rotation else np.pi
    self.ref_rotation_criterion = get_rotation_loss(config.ref_rotation_loss)(
        config.num_ref_rotation_bins, activation='tanh', min_angle=min_angle, max_angle=max_angle)

  def initialize_models(self, config, dataset):
    self.anchor_ratios = np.reshape(self.config.rpn_anchor_ratios, (-1, 3))
    backbone_model_class = models.load_model(config.backbone_model)
    self.backbone = backbone_model_class(dataset.NUM_IN_CHANNEL, config).to(self.device)
    self.feature_upsample_network = self.FEATURE_UPSAMPLE_NETWORK_MODEL(
        self.backbone.out_channels, self.backbone.OUT_PIXEL_DIST, config).to(self.device)
    assert tuple(self.config.rpn_strides) == self.backbone.OUT_PIXEL_DIST
    assert tuple(self.config.rpn_strides) == self.feature_upsample_network.OUT_PIXEL_DIST
    self.region_proposal_network = self.REGION_PROPOSAL_NETWORK_MODEL(
        self.feature_upsample_network.out_channels, self.anchor_ratios.shape[0], config,
        dataset.IS_ROTATION_BBOX, self.rpn_rotation_criterion, dataset.NUM_LABELS).to(self.device)
    if not self.train_rpnonly:
      self.fpn_classifier_network = self.REGION_REFINEMENT_NETWORK_MODEL(
          self.feature_upsample_network.out_channels, self.feature_upsample_network.OUT_PIXEL_DIST,
          dataset.NUM_LABELS, config, dataset.IS_ROTATION_BBOX,
          self.ref_rotation_criterion).to(self.device)

  def initialize_optimizer(self, config):
    rpnonly_params = itertools.chain(
        self.backbone.parameters(),
        self.feature_upsample_network.parameters(),
        self.region_proposal_network.parameters())
    return {
        'default': solvers.initialize_optimizer(self.parameters(), config),
        'rpnonly': solvers.initialize_optimizer(rpnonly_params, config),
    }

  def step_optimizer(self, loss, optimizers, schedulers, iteration):
    def _layer_requires_grad(layer, requires_grad):
      for param in layer.parameters():
        param.requires_grad = requires_grad

    # Check if loss is nan.
    if not torch.isfinite(loss['loss']):
      logging.warning('Encountered invalid loss. Ignoring this training sample.')
      self.reset_gradient(optimizers)
      return

    # Stop gradient on particular layers based on freezer policy.
    freezer_schedule = None
    if self.config.freezer.lower() == 'detectron':
      freezer_schedule = (0.125, 0.5)
    elif self.config.freezer.lower() == 'detectron_light':
      freezer_schedule = (0.05, 0.2)
    elif self.config.freezer.lower() != 'none':
      raise ValueError(f'Unknown freezer policy {self.config.freezer}')
    if freezer_schedule is not None:
      if iteration < self.config.max_iter * freezer_schedule[0]:
        _layer_requires_grad(self.backbone, False)
      elif iteration < self.config.max_iter * freezer_schedule[1]:
        _layer_requires_grad(self.backbone, False)
        _layer_requires_grad(self.backbone.layer3, True)
        _layer_requires_grad(self.backbone.layer4, True)
      else:
        _layer_requires_grad(self.backbone, True)

    assert set(optimizers) == set(schedulers)
    if loss.get('optimize_ref', False):
      optimizers['default'].step()
    else:
      if not self.config.train_skip_rpnonly:
        optimizers['rpnonly'].step()
    schedulers['default'].step()
    schedulers['rpnonly'].step()

  def _get_detection_target(self, b_proposals, b_rotations, b_gt_classes, b_gt_boxes,
                            b_gt_rotations):
    def _random_subsample_idxs(indices, num_samples):
      if indices.size(0) > num_samples:
        return torch.from_numpy(np.random.choice(indices.cpu().numpy(), num_samples,
                                replace=False)).to(indices)
      return indices
    b_rois, b_roi_gt_classes, b_deltas, b_roi_gt_box_assignment = [], [], [], []
    b_rots, b_rot_deltas = (None, None) if b_rotations is None else ([], [])
    for i, (proposals, gt_classes, gt_boxes) in enumerate(
            zip(b_proposals, b_gt_classes, b_gt_boxes)):
      with torch.no_grad():
        proposals = proposals[~torch.all(proposals == 0, 1)]
        gt_boxes = torch.from_numpy(gt_boxes).to(proposals)
        if gt_boxes.shape[0] == 0 or proposals.shape[0] == 0:
          b_deltas.append(torch.zeros((gt_boxes.shape[0], proposals.shape[1])).to(proposals))
          b_roi_gt_classes.append(torch.zeros(0).to(proposals).long())
          b_rois.append(torch.zeros((gt_boxes.shape[0], proposals.shape[1])).to(proposals))
          if b_rotations is not None:
            b_rots.append(torch.zeros(0).to(proposals))
            b_rot_deltas.append(torch.zeros(0).to(proposals))
          continue
        gt_classes = torch.from_numpy(gt_classes).to(proposals)
        pred_rotation, gt_rotation = None, None
        if self.config.ref_rotation_overlap and b_rotations is not None:
          pred_rotation = b_rotations[i]
          gt_rotation = b_gt_rotations[i]
        overlaps = detection_utils.compute_overlaps(proposals, gt_boxes, pred_rotation, gt_rotation)
        roi_iou_max = overlaps.max(1)[0]
        positive_roi = roi_iou_max > self.config.detection_match_positive_iou_threshold
        positive_indices = torch.where(positive_roi)[0]
        if self.config.force_proposal_match:
          torch.unique(torch.cat((torch.argmax(overlaps, 0), positive_indices)))
        negative_indices = torch.where(~positive_roi)[0]
        positive_count = int(self.config.roi_num_proposals_training
                             * self.config.roi_positive_ratio_training)
        positive_indices = _random_subsample_idxs(positive_indices, positive_count)
        positive_count = positive_indices.size(0)
        negative_count = int(
            positive_count / self.config.roi_positive_ratio_training) - positive_count
        negative_indices = _random_subsample_idxs(negative_indices, negative_count)
        negative_count = negative_indices.size(0)

        positive_rois = torch.index_select(proposals, 0, positive_indices)
        negative_rois = torch.index_select(proposals, 0, negative_indices)

        positive_overlaps = torch.index_select(overlaps, 0, positive_indices)
        roi_gt_box_assignment = (
            positive_overlaps.argmax(1)
            if positive_count else torch.empty(0).to(positive_overlaps).long())
        b_roi_gt_box_assignment.append(roi_gt_box_assignment)

        roi_gt_boxes = torch.index_select(gt_boxes, 0, roi_gt_box_assignment)
        roi_gt_classes = torch.index_select(gt_classes, 0, roi_gt_box_assignment).long()
        deltas = detection_utils.get_bbox_target(
            positive_rois, roi_gt_boxes, self.config.rpn_bbox_std)
        if b_rotations is not None:
          rotation = torch.cat((b_rotations[i][positive_indices], b_rotations[i][negative_indices]))
          if self.config.normalize_rotation2:
            rotation = rotation / 2 + np.pi / 2
          b_rots.append(rotation)
          b_rot_deltas.append(
              detection_utils.normalize_rotation(
                  torch.from_numpy(b_gt_rotations[i]).to(proposals)[roi_gt_box_assignment]
                  - b_rotations[i][positive_indices]))

        b_deltas.append(deltas)
        roi_gt_classes = torch.cat(
            (roi_gt_classes + 1, torch.zeros(negative_count).to(roi_gt_classes)))
        b_roi_gt_classes.append(roi_gt_classes)

        rois = torch.cat((positive_rois, negative_rois))
        b_rois.append(rois)
    return b_rois, b_rots, b_roi_gt_classes, b_deltas, b_rot_deltas, b_roi_gt_box_assignment

  def _get_mask_target(self, batch_bboxes, batch_coords, batch_bboxes_id=None,
                       batch_instance_target=None, is_train=True):
    return None, None, None

  def detection_refinement(self, b_probs, b_rois, b_deltas, b_rots, b_rotdeltas):
    if b_probs is None:
      num_channel = 9 if b_rots is None else 8
      return [np.zeros((0, num_channel))]
    num_batch = [rois.shape[0] for rois in b_rois]
    num_samples = sum(num_batch)
    assert num_samples == b_probs.shape[0] == b_deltas.shape[0]
    if b_rots is not None:
      assert num_samples == sum(rots.shape[0] for rots in b_rots) == b_rotdeltas.shape[0]
    batch_split = [(sum(num_batch[:i]), sum(num_batch[:(i + 1)])) for i in range(len(num_batch))]
    b_probs = [b_probs[i:j] for (i, j) in batch_split]
    b_deltas = [b_deltas[i:j] for (i, j) in batch_split]
    if b_rots is not None:
      b_rotdeltas = [b_rotdeltas[i:j] for (i, j) in batch_split]
    b_nms = []
    b_nms_rot = None if b_rots is None else []
    for i, (probs, rois, deltas) in enumerate(zip(b_probs, b_rois, b_deltas)):
      rois = rois.reshape(-1, rois.shape[-1])
      class_ids = torch.argmax(probs, dim=1)
      batch_slice = range(probs.shape[0])
      class_scores = probs[batch_slice, class_ids]
      class_deltas = deltas[batch_slice, class_ids - 1]
      class_deltas *= torch.tensor(self.config.rpn_bbox_std).to(deltas)
      refined_rois = detection_utils.apply_box_deltas(rois, class_deltas,
                                                      self.config.normalize_bbox)
      if b_rots is not None:
        class_rot_deltas = b_rotdeltas[i][batch_slice, class_ids - 1]
        class_rot_deltas = self.ref_rotation_criterion.pred(class_rot_deltas)
        refined_rots = detection_utils.normalize_rotation(b_rots[i] + class_rot_deltas)
      keep = torch.where(class_ids > 0)[0].cpu().numpy()
      if self.config.detection_min_confidence:
        conf_keep = torch.where(class_scores > self.config.detection_min_confidence)[0]
        keep = np.array(list(set(conf_keep.cpu().numpy()).intersection(keep)))
      if keep.size == 0:
        b_nms.append(np.zeros((0, 8)))
        if b_rots is not None:
          b_nms_rot.append(np.zeros(0))
      else:
        pre_nms_class_ids = class_ids[keep] - 1
        pre_nms_scores = class_scores[keep]
        pre_nms_rois = refined_rois[keep]
        if b_rots is not None:
          pre_nms_rots = refined_rots[keep]
        nms_scores = []
        nms_rois = []
        nms_classes = []
        nms_rots = []
        for class_id in torch.unique(pre_nms_class_ids):
          class_nms_mask = pre_nms_class_ids == class_id
          class_nms_scores = pre_nms_scores[class_nms_mask]
          class_nms_rois = pre_nms_rois[class_nms_mask]
          pre_nms_class_rots = None
          if b_rots is not None:
            pre_nms_class_rots = pre_nms_rots[class_nms_mask]
          nms_roi, nms_rot, nms_score = detection_utils.non_maximum_suppression(
              class_nms_rois, pre_nms_class_rots, class_nms_scores,
              self.config.detection_nms_threshold, self.config.detection_max_instances,
              self.config.detection_rot_nms, self.config.detection_aggregate_overlap)
          nms_rois.append(nms_roi)
          nms_scores.append(nms_score)
          nms_classes.append(torch.ones(len(nms_score)).to(class_nms_rois) * class_id)
          if b_rots is not None:
            if self.config.normalize_rotation2:
              nms_rot = nms_rot / 2 + np.pi / 2
            nms_rots.append(nms_rot)
        nms_scores = torch.cat(nms_scores)
        nms_rois = torch.cat(nms_rois)
        nms_classes = torch.cat(nms_classes)
        detection_max_instances = min(self.config.detection_max_instances, nms_scores.shape[0])
        ix = torch.topk(nms_scores, detection_max_instances)[1]
        nms_rois_unnorm = detection_utils.unnormalize_boxes(
            nms_rois[ix].cpu().numpy(), self.config.max_ptc_size)
        nms_bboxes = np.hstack((nms_rois_unnorm, nms_classes[ix, None].cpu().numpy(),
                                nms_scores[ix, None].cpu().numpy()))
        if b_rots is not None:
          nms_rots = torch.cat(nms_rots)[ix, None].cpu().numpy()
          nms_bboxes = np.hstack((nms_bboxes[:, :6], nms_rots, nms_bboxes[:, 6:]))
        b_nms.append(nms_bboxes)
    return b_nms

  def _get_rpnonly_detection(self, rpn_rois, rpn_rois_score, rpn_rois_rotation):
    detections = []
    if rpn_rois is not None:
      for i, (rpn_roi, rpn_score) in enumerate(zip(rpn_rois, rpn_rois_score)):
        rpn_roi_unnorm = detection_utils.unnormalize_boxes(
            rpn_roi[:rpn_score.shape[0], :6].cpu().numpy(), self.config.max_ptc_size)
        if self.EVAL_PERCLASS_MAP:
          rpn_classes = rpn_roi[:, 6, None].cpu().numpy()
        else:
          rpn_classes = np.zeros((rpn_roi.shape[0], 1))
        detection = np.hstack((rpn_roi_unnorm, rpn_classes, rpn_score.cpu().numpy()[:, None]))
        if rpn_rois_rotation is not None:
          detection_rot = rpn_rois_rotation[i].cpu().numpy()[:, None]
          if self.config.normalize_rotation2:
            detection_rot = detection_rot / 2 + np.pi / 2
          detection = np.hstack((detection[:, :6], detection_rot, detection[:, 6:]))
        detection_mask = detection[:, -1] > self.config.post_nms_min_confidence
        detection = detection[detection_mask]
        detections.append(detection)
    else:
      detections = [np.zeros((0, 9 if self.dataset.IS_ROTATION_BBOX else 8))]
    return detections

  def forward(self, datum, is_train):
    backbone_outputs = self.backbone(datum['sinput'])
    output, generate_proposal = self.forward_region_proposal(datum, backbone_outputs, is_train)
    if generate_proposal and not self.train_rpnonly:
      refinement_output = self.forward_refinement_head(
          datum, output['rpn_rois'], output['rpn_rois_rotation'], output['fpn_outputs'], is_train)
      output.update(refinement_output)
    elif self.train_rpnonly and not is_train:
      output['detection'] = self._get_rpnonly_detection(output['rpn_rois'],
                                                        output['rpn_rois_score'],
                                                        output['rpn_rois_rotation'])

    return output

  def forward_refinement_head(self, datum, rpn_rois, rpn_rois_rotation, fpn_outputs, is_train):
    if is_train:
      ref_rois, ref_rots, target_class_ids, target_bbox, target_rots, roi_gt_box_assignment \
          = self._get_detection_target(rpn_rois, rpn_rois_rotation, datum['bboxes_cls'],
                                       datum['bboxes_normalized'], datum['bboxes_rotations'])
      unnorm_ref_rois = [
          detection_utils.unnormalize_boxes(roi.cpu().numpy(), self.config.max_ptc_size)
          for roi in ref_rois
      ]
      unnorm_ref_posrois = [r[:len(b)] for r, b in zip(unnorm_ref_rois, roi_gt_box_assignment)]
      mask_coords, mask_insts, _ = self._get_mask_target(unnorm_ref_posrois, datum['coords'],
                                                         roi_gt_box_assignment,
                                                         datum['instance_target'], is_train=True)
      ref_class_logits, ref_class_probs, ref_bbox, ref_rotdelta = self.fpn_classifier_network(
          unnorm_ref_rois, ref_rots, fpn_outputs)
      output = {
          'ref_rois': ref_rois,
          'ref_rots': ref_rots,
          'unnorm_ref_rois': unnorm_ref_rois,
          'unnorm_ref_posrois': unnorm_ref_posrois,
          'ref_target_class_ids': target_class_ids,
          'ref_target_bbox': target_bbox,
          'ref_target_rots': target_rots,
          'ref_mask_coords': mask_coords,
          'ref_mask_insts': mask_insts,
          'ref_class_logits': ref_class_logits,
          'ref_class_probs': ref_class_probs,
          'ref_bbox': ref_bbox,
          'ref_rotdelta': ref_rotdelta,
      }
    else:
      rpn_rois = [roi[~torch.all(roi == 0, 1)] for roi in rpn_rois]
      unnorm_ref_rois = [
          detection_utils.unnormalize_boxes(roi.cpu().numpy(), self.config.max_ptc_size)
          for roi in rpn_rois
      ]
      ref_rots = rpn_rois_rotation
      if ref_rots is not None and self.config.normalize_rotation2:
        ref_rots = [rot / 2 + np.pi / 2 for rot in ref_rots]
      _, ref_class_probs, ref_bbox, ref_rotdelta = self.fpn_classifier_network(
          unnorm_ref_rois, ref_rots, fpn_outputs)
      detection = self.detection_refinement(
          ref_class_probs, rpn_rois, ref_bbox, rpn_rois_rotation, ref_rotdelta)
      mask_coords, _, mask_coordmasks = self._get_mask_target(detection, datum['coords'],
                                                              is_train=False)
      output = {
          'detection': detection,
          'mask_coords': mask_coords,
          'mask_coordmasks': mask_coordmasks,
      }
    return output

  def loss(self, datum, output):
    rpn_loss, rpn_loss_details = self.loss_region_proposal(datum, output)
    refienment_loss, refinement_loss_details = self.loss_refinement_head(datum, output)
    loss = rpn_loss + refienment_loss

    losses = {
        **rpn_loss_details,
        **refinement_loss_details,
        'loss': loss,
    }

    return losses

  def loss_refinement_head(self, datum, output):
    if self.train_rpnonly:
      return torch.scalar_tensor(0.).to(self.device), dict()

    optimize_ref = False
    if output.get('ref_class_logits') is not None:
      if output['ref_class_logits'].shape[0] \
         > self.config.train_ref_min_sample_per_batch * self.config.batch_size:
        optimize_ref = True
      ref_labels = torch.cat(output['ref_target_class_ids'])
      ref_class_loss = self.ref_class_criterion(output['ref_class_logits'], ref_labels)
    else:
      ref_class_loss = torch.scalar_tensor(0.).to(self.device)

    if output.get('ref_bbox') is not None:
      positive_rois = torch.where(ref_labels > 0)[0]
      bbox_idx = ref_labels[positive_rois] - 1
      ref_bbox_pred = output['ref_bbox'][positive_rois, bbox_idx]
      ref_bbox_gt = torch.cat(output['ref_target_bbox'])
      ref_bbox_loss = self.ref_bbox_criterion(ref_bbox_pred, ref_bbox_gt)
    else:
      ref_bbox_loss = torch.scalar_tensor(0.).to(self.device)

    ref_rotation_diff = torch.scalar_tensor(0.).to(self.device)
    train_rotation = output.get('ref_rotdelta') is not None
    if train_rotation:
      ref_rotation_pred = output['ref_rotdelta'][positive_rois, bbox_idx]
      ref_rotation_gt = torch.cat(output['ref_target_rots'])
      ref_rotation_loss = self.ref_rotation_criterion(ref_rotation_pred, ref_rotation_gt)
      ref_rotation_pred = self.ref_rotation_criterion.pred(ref_rotation_pred)
      ref_rotation_diff = torch.abs(
          detection_utils.normalize_rotation(ref_rotation_pred - ref_rotation_gt)).mean()
    else:
      ref_rotation_loss = torch.scalar_tensor(0.).to(self.device)

    loss = torch.scalar_tensor(0.).to(self.device)
    if optimize_ref:
      loss += self.config.ref_class_weight * ref_class_loss \
          + self.config.ref_bbox_weight * ref_bbox_loss
      if train_rotation:
        loss += self.config.ref_rotation_weight * ref_rotation_loss

    loss_details = {
        'ref_class_loss': ref_class_loss,
        'ref_bbox_loss': ref_bbox_loss,
        'optimize_ref': torch.tensor([optimize_ref], dtype=torch.int).to(loss)[0],
    }

    if train_rotation:
      loss_details['ref_rotation_loss'] = ref_rotation_loss
      loss_details['ref_rotation_diff'] = ref_rotation_diff

    return loss, loss_details


class FasterRCNN(FasterRCNNBase):

  FEATURE_UPSAMPLE_NETWORK_MODEL = models.detection.FeatureUpsampleNetwork
  REGION_PROPOSAL_NETWORK_MODEL = models.detection.RegionProposalNetwork
  REGION_REFINEMENT_NETWORK_MODEL = models.detection.FeaturePyramidClassifierNetwork

  def __init__(self, config, dataset):
    super().__init__(config, dataset)

  def forward_region_proposal(self, datum, backbone_outputs, is_train):
    fpn_outputs = self.feature_upsample_network(backbone_outputs)
    detection_utils.check_backbone_shapes(datum['backbone_shapes'], fpn_outputs)
    num_proposals = self.config.rpn_num_proposals_training if is_train \
        else self.config.rpn_num_proposals_inference
    generate_proposal = not (is_train and self.train_rpnonly)
    rpn_class_logits, rpn_bbox, rpn_rotation, rpn_rois, rpn_rois_rotation, rpn_rois_score = \
        self.region_proposal_network(fpn_outputs, datum['anchors'], num_proposals,
                                     generate_proposal=generate_proposal)
    output = {
        'fpn_outputs': fpn_outputs,
        'rpn_class_logits': rpn_class_logits,
        'rpn_bbox': rpn_bbox,
        'rpn_rotation': rpn_rotation,
        'rpn_rois': rpn_rois,
        'rpn_rois_rotation': rpn_rois_rotation,
        'rpn_rois_score': rpn_rois_score,
    }
    return output, generate_proposal

  def loss_region_proposal(self, datum, output):
    # Compute RPN class loss.
    rpn_match_gt = datum['rpn_match'].flatten()
    rpn_match_mask = rpn_match_gt.nonzero().squeeze()
    if rpn_match_mask.size(0) > 0:
      rpn_match_gt_valid = (rpn_match_gt[rpn_match_mask] == 1).long()
      rpn_class_logits = output['rpn_class_logits'].reshape(-1, 2)[rpn_match_mask]
      rpn_class_loss = self.rpn_class_criterion(rpn_class_logits, rpn_match_gt_valid)
    else:
      rpn_class_loss = torch.scalar_tensor(0.).to(self.device)

    # Compute RPN bbox regression loss.
    rpn_bbox_mask = torch.where(rpn_match_gt == 1)[0]
    if rpn_bbox_mask.size(0) > 0:
      rpn_bbox_pred = output['rpn_bbox'].reshape(-1, 6)[rpn_bbox_mask]
      rpn_bbox_gt = datum['rpn_bbox'].reshape(-1, 6)
      rpn_gt_mask = ~torch.all(rpn_bbox_gt == 0, 1)
      rpn_bbox_gt = rpn_bbox_gt[rpn_gt_mask]
      rpn_bbox_loss = self.rpn_bbox_criterion(rpn_bbox_pred, rpn_bbox_gt)
    else:
      rpn_bbox_loss = torch.scalar_tensor(0.).to(self.device)

    loss = self.config.rpn_class_weight * rpn_class_loss \
        + self.config.rpn_bbox_weight * rpn_bbox_loss

    # Compute RPN bbox rotation loss.
    rpn_rotation_diff = torch.scalar_tensor(0.).to(self.device)
    if output.get('rpn_rotation') is not None:
      if rpn_bbox_mask.size(0) > 0:
        rpn_rotation_pred = output['rpn_rotation'].reshape(
            -1, self.rpn_rotation_criterion.NUM_OUTPUT)[rpn_bbox_mask]
        rpn_rotation_gt = datum['rpn_rotation'].flatten()[rpn_gt_mask]
        rpn_rotation_loss = self.rpn_rotation_criterion(rpn_rotation_pred, rpn_rotation_gt)
        rpn_rotation_pred = self.rpn_rotation_criterion.pred(rpn_rotation_pred)
        rpn_rotation_diff = torch.abs(
            detection_utils.normalize_rotation(rpn_rotation_pred - rpn_rotation_gt)).mean()
      else:
        rpn_rotation_loss = torch.scalar_tensor(0.).to(self.device)
      loss += self.config.rpn_rotation_weight * rpn_rotation_loss

    loss_details = {
        'rpn_class_loss': rpn_class_loss,
        'rpn_bbox_loss': rpn_bbox_loss,
    }

    if output.get('rpn_rotation') is not None:
      loss_details['rpn_rotation_loss'] = rpn_rotation_loss
      loss_details['rpn_rotation_diff'] = rpn_rotation_diff

    return loss, loss_details


class SparseGenerativeFasterRCNN(FasterRCNNBase):

  FEATURE_UPSAMPLE_NETWORK_MODEL = models.detection.SparseGenerativeFeatureUpsampleNetwork
  REGION_PROPOSAL_NETWORK_MODEL = models.detection.SparseRegionProposalNetwork
  REGION_REFINEMENT_NETWORK_MODEL = models.detection.SparseFeaturePyramidClassifierNetwork

  def __init__(self, config, dataset):
    super().__init__(config, dataset)
    if not config.load_sparse_gt_data:
      raise ValueError("Sparse generation requires full anchor information."
                       "Set `--load_sparse_gt_data true`")

  def visualize_groundtruth(self, datum, iteration):
    # TODO: Implement ground-truth visualization.
    return

  def visualize_predictions(self, datum, output, iteration):
    # FIXME: Refactor visualization code.
    coords = datum['coords'].numpy()
    batch_size = coords[:, 0].max() + 1
    output_path = pathlib.Path(self.config.visualize_path)
    output_path.mkdir(exist_ok=True)
    for i in range(batch_size):
      # Visualize RGB input.
      coords_mask = coords[:, 0] == i
      coords_b = coords[coords_mask, 1:]
      if datum['input'].shape[1] > 3:
        rgb_b = ((datum['input'][:, :3] + 0.5) * 255).numpy()
      else:
        rgb_b = ((datum['input'].repeat(1, 3) - datum['input'].min())
                 / (datum['input'].max() - datum['input'].min()) * 255).numpy()
      ptc = np.hstack((coords_b, rgb_b))
      bbox_scores = output['detection'][i][:, -1]
      bbox_mask = bbox_scores > self.config.visualize_min_confidence
      if self.dataset.IS_ROTATION_BBOX:
        bboxes = pc_utils.visualize_bboxes(output['detection'][i][bbox_mask][:, :7],
                                           output['detection'][i][bbox_mask][:, 7],
                                           bbox_param='xyzxyzr')
      else:
        bboxes = pc_utils.visualize_bboxes(output['detection'][i][bbox_mask][:, :6],
                                           output['detection'][i][bbox_mask][:, 6])
      pc_utils.visualize_pcd(ptc, bboxes)

  def initialize_losses(self, config):
    super().initialize_losses(config)
    self.sfpn_class_criterion = get_classification_loss(config.sfpn_classification_loss)(
        ignore_index=config.ignore_label)

  def forward_region_proposal(self, datum, backbone_outputs, is_train):
    fpn_outputs, fpn_targets, fpn_classification = self.feature_upsample_network(
        backbone_outputs, datum['anchor_match_coords'], is_train)
    num_proposals = self.config.rpn_num_proposals_training if is_train \
        else self.config.rpn_num_proposals_inference
    generate_proposal = not (is_train and self.train_rpnonly)
    rpn_class_logits, rpn_bbox, rpn_rotation, rpn_rois, rpn_rois_rotation, rpn_rois_score, \
        rpn2anchor_maps = self.region_proposal_network(fpn_outputs, datum['sparse_anchor_coords'],
                                                       datum['sparse_anchor_centers'],
                                                       num_proposals,
                                                       generate_proposal=generate_proposal)
    output = {
        'fpn_outputs': fpn_outputs,
        'fpn_targets': fpn_targets,
        'fpn_classification': fpn_classification,
        'rpn_class_logits': rpn_class_logits,
        'rpn_bbox': rpn_bbox,
        'rpn_rotation': rpn_rotation,
        'rpn_rois': rpn_rois,
        'rpn_rois_rotation': rpn_rois_rotation,
        'rpn_rois_score': rpn_rois_score,
        'rpn2anchor_maps': rpn2anchor_maps,
    }
    return output, generate_proposal

  def loss_region_proposal(self, datum, output):
    if output['fpn_targets'] is not None:
      num_layers = len(self.backbone.OUT_PIXEL_DIST)
      sfpn_class_loss = torch.scalar_tensor(0.).to(self.device)
      for logits, targets in zip(output['fpn_classification'], output['fpn_targets']):
        if targets is None:
          continue
        sfpn_class_loss += self.sfpn_class_criterion(logits.F, targets.to(self.device)) / num_layers

    if output.get('rpn2anchor_maps') is None:
      output['rpn2anchor_maps'] = []
      for i, (rpn_class_logits, anchor) in enumerate(
              zip(output['rpn_class_logits'], datum['sparse_anchor_coords'])):
        if rpn_class_logits is None:
          output['rpn2anchor_maps'].append((None, None))
        else:
          assert rpn_class_logits.coords_key == output['rpn_bbox'][i].coords_key
          if output['rpn_rotation'][i] is not None:
            assert rpn_class_logits.coords_key == output['rpn_rotation'][i].coords_key
          output['rpn2anchor_maps'].append(detection_utils.map_coordinates(
              rpn_class_logits, anchor, check_input_map=True))

    rpn_masks = []
    rpn_class_preds = []
    rpn_class_gts = []
    for rpn_class_logits, rpn_match, (rpn2anchor, anchor2rpn) in zip(
            output['rpn_class_logits'], datum['sparse_rpn_match'], output['rpn2anchor_maps']):
      if rpn_class_logits is None:
        rpn_masks.append(None)
        continue
      rpn_match = rpn_match[anchor2rpn].flatten()
      rpn_masks.append(torch.where(rpn_match == 1)[0])
      rpn_match_mask = rpn_match.nonzero().squeeze(-1)
      if rpn_match_mask.size(0) > 0:
        rpn_class_preds.append(rpn_class_logits.F[rpn2anchor].reshape(-1, 2)[rpn_match_mask])
        rpn_class_gts.append((rpn_match[rpn_match_mask] == 1).long().to(self.device))
    rpn_class_loss = torch.scalar_tensor(0.).to(self.device)
    if rpn_class_preds:
      rpn_class_loss = self.rpn_class_criterion(torch.cat(rpn_class_preds),
                                                torch.cat(rpn_class_gts))

    rpn_bbox_preds = []
    rpn_bbox_gts = []
    for rpn_bbox_pred, rpn_bbox_gt, rpn_mask, (rpn2anchor, anchor2rpn) in zip(
            output['rpn_bbox'], datum['sparse_rpn_bbox'], rpn_masks, output['rpn2anchor_maps']):
      if rpn_bbox_pred is not None and rpn_mask.size(0) > 0:
        rpn_bbox_preds.append(rpn_bbox_pred.F[rpn2anchor].reshape(-1, 6)[rpn_mask])
        rpn_bbox_gts.append(rpn_bbox_gt[anchor2rpn].reshape(-1, 6)[rpn_mask].to(rpn_bbox_pred.F))
    rpn_bbox_loss = torch.scalar_tensor(0.).to(self.device)
    if rpn_bbox_preds:
      rpn_bbox_loss = self.rpn_bbox_criterion(torch.cat(rpn_bbox_preds), torch.cat(rpn_bbox_gts))

    loss = self.config.rpn_class_weight * rpn_class_loss \
        + self.config.rpn_bbox_weight * rpn_bbox_loss

    if output['fpn_targets'] is not None:
      loss += self.config.sfpn_class_weight * sfpn_class_loss

    train_rotation = any([rot is not None for rot in output.get('rpn_rotation')])
    if train_rotation:
      rpn_rotation_preds = []
      rpn_rotation_gts = []
      for rpn_rotation_pred, rpn_rotation_gt, rpn_mask, (rpn2anchor, anchor2rpn) in zip(
              output['rpn_rotation'], datum['sparse_rpn_rotation'], rpn_masks,
              output['rpn2anchor_maps']):
        if rpn_rotation_pred is not None and rpn_mask.size(0) > 0:
          rpn_rotation_preds.append(rpn_rotation_pred.F[rpn2anchor].reshape(
              -1, self.rpn_rotation_criterion.NUM_OUTPUT)[rpn_mask])
          rpn_rotation_gts.append(
              rpn_rotation_gt[anchor2rpn].flatten()[rpn_mask].to(rpn_rotation_pred.F))
      rpn_rotation_loss = torch.scalar_tensor(0.).to(self.device)
      rpn_rotation_diff = torch.scalar_tensor(0.).to(self.device)
      if rpn_rotation_preds:
        rpn_rotation_preds = torch.cat(rpn_rotation_preds)
        rpn_rotation_gts = torch.cat(rpn_rotation_gts)
        rpn_rotation_loss = self.rpn_rotation_criterion(rpn_rotation_preds, rpn_rotation_gts)
        rpn_rotation_preds = self.rpn_rotation_criterion.pred(rpn_rotation_preds)
        rpn_rotation_diff = torch.abs(detection_utils.normalize_rotation(
            rpn_rotation_preds - rpn_rotation_gts)).mean()
      loss += self.config.rpn_rotation_weight * rpn_rotation_loss

    loss_details = {
        'rpn_class_loss': rpn_class_loss,
        'rpn_bbox_loss': rpn_bbox_loss,
    }

    if output['fpn_targets'] is not None:
      loss_details['sfpn_class_loss'] = sfpn_class_loss

    if train_rotation:
      loss_details['rpn_rotation_loss'] = rpn_rotation_loss
      loss_details['rpn_rotation_diff'] = rpn_rotation_diff

    return loss, loss_details


class SparseGenerativeOneShotDetector(SparseGenerativeFasterRCNN):

  REGION_PROPOSAL_NETWORK_MODEL = models.detection.SparseRegionProposalClassifierNetwork
  REGION_REFINEMENT_NETWORK_MODEL = None

  def __init__(self, config, dataset):
    super().__init__(config, dataset)
    self.EVAL_PERCLASS_MAP = True

  def initialize_losses(self, config):
    super().initialize_losses(config)
    self.rpn_semantic_criterion = get_classification_loss(config.rpn_semantic_loss)(
        ignore_index=config.ignore_label)

  def forward_region_proposal(self, datum, backbone_outputs, is_train):
    fpn_outputs, fpn_targets, fpn_classification = self.feature_upsample_network(
        backbone_outputs, datum['anchor_match_coords'], is_train)
    num_proposals = self.config.rpn_num_proposals_training if is_train \
        else self.config.rpn_num_proposals_inference
    generate_proposal = not (is_train and self.train_rpnonly)
    rpn_class_logits, rpn_semantic_logits, rpn_bbox, rpn_rotation, rpn_rois, rpn_rois_rotation, \
        rpn_rois_score = self.region_proposal_network(
            fpn_outputs, num_proposals, generate_proposal=generate_proposal)
    output = {
        'fpn_targets': fpn_targets,
        'fpn_classification': fpn_classification,
        'rpn_class_logits': rpn_class_logits,
        'rpn_semantic_logits': rpn_semantic_logits,
        'rpn_bbox': rpn_bbox,
        'rpn_rotation': rpn_rotation,
        'rpn_rois': rpn_rois,
        'rpn_rois_rotation': rpn_rois_rotation,
        'rpn_rois_score': rpn_rois_score,
    }
    return output, generate_proposal

  def loss_region_proposal(self, datum, output):
    loss, loss_details = super().loss_region_proposal(datum, output)

    rpn_semantic_preds = []
    rpn_semantic_gts = []
    for rpn_semantic_logits, rpn_semantic_gt, rpn_match, (rpn2anchor, anchor2rpn) in zip(
            output['rpn_semantic_logits'], datum['sparse_rpn_cls'], datum['sparse_rpn_match'],
            output['rpn2anchor_maps']):
      if rpn_semantic_logits is None:
        continue
      rpn_mask = torch.where(rpn_match[anchor2rpn].flatten() == 1)[0]
      if rpn_mask.size(0) > 0:
        rpn_semantic_preds.append(
            rpn_semantic_logits.F[rpn2anchor].reshape(-1, self.dataset.NUM_LABELS)[rpn_mask])
        rpn_semantic_gts.append(rpn_semantic_gt[anchor2rpn].flatten()[rpn_mask].to(self.device))
    rpn_semantic_loss = torch.scalar_tensor(0.).to(self.device)
    if rpn_semantic_preds:
      rpn_semantic_loss = self.rpn_semantic_criterion(torch.cat(rpn_semantic_preds),
                                                      torch.cat(rpn_semantic_gts).long())

    loss += self.config.rpn_semantic_weight * rpn_semantic_loss
    loss_details['rpn_semantic_loss'] = rpn_semantic_loss

    return loss, loss_details


class SparseEncoderOnlyOneShotDetector(SparseGenerativeOneShotDetector):

  FEATURE_UPSAMPLE_NETWORK_MODEL = models.detection.FakeSparseGenerativeFeatureUpsampleNetwork


class SparseNoPruningOneShotDetector(SparseGenerativeOneShotDetector):

  FEATURE_UPSAMPLE_NETWORK_MODEL = models.detection.SparseGenerativeNoPruningFeatureUpsampleNetwork
