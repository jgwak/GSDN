# Majority of the code from https://github.com/Yang7879/3D-BoNet
# MIT License, Copyright (c) 2019 Bo Yang

import glob
import os

import h5py
import numpy as np
import scipy.io
import scipy.stats
import torch
import tqdm

from lib.detection_ap import DetectionAPCalculator
from lib.detection_utils import compute_overlaps, non_maximum_suppression
from lib.pc_utils import get_bbox, bboxes2corners
from lib.utils import log_meters


DATASET_PATH = '/home/jgwak/Downloads/bonet_s3dis/'
TRAIN_AREA = ('Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6')
RESULT_PATH = '/scr/050_Area_5/'
DATA_SEM_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
NMS_THRES = 0.35


def get_mean_insSize_by_sem(dataset_path, train_areas):

  mean_insSize_by_sem = {}
  for sem in DATA_SEM_IDS:
    mean_insSize_by_sem[sem] = []

  for a in train_areas:
    print('get mean insSize, check train area:', a)
    files = sorted(glob.glob(dataset_path + a + '*.h5'))
    for file_path in files:
      fin = h5py.File(file_path, 'r')
      semIns_labels = fin['labels'][:].reshape([-1, 2])
      ins_labels = semIns_labels[:, 1]
      sem_labels = semIns_labels[:, 0]

      ins_idx = np.unique(ins_labels)
      for ins_id in ins_idx:
        tmp = (ins_labels == ins_id)
        sem = scipy.stats.mode(sem_labels[tmp])[0][0]
        mean_insSize_by_sem[sem].append(np.sum(np.asarray(tmp, dtype=np.float32)))

  for sem in mean_insSize_by_sem:
    mean_insSize_by_sem[sem] = np.mean(mean_insSize_by_sem[sem])

  return mean_insSize_by_sem


def get_sem_for_ins(ins_by_pts, sem_by_pts):
  ins_cls_dic = {}
  ins_idx, cnt = np.unique(ins_by_pts, return_counts=True)
  for ins_id, cn in zip(ins_idx, cnt):
    if ins_id == -1:
      continue  # empty ins
    temp = sem_by_pts[np.argwhere(ins_by_pts == ins_id)][:, 0]
    sem_for_this_ins = scipy.stats.mode(temp)[0][0]
    ins_cls_dic[ins_id] = sem_for_this_ins
  return ins_cls_dic


def block_merge(volume, volume_seg, pts, grouplabel, groupseg, gap=1e-3):
  overlapgroupcounts = np.zeros([100, 1000])
  groupcounts = np.ones(100)
  x = (pts[:, 0] / gap).astype(np.int32)
  y = (pts[:, 1] / gap).astype(np.int32)
  z = (pts[:, 2] / gap).astype(np.int32)
  for i in range(pts.shape[0]):
    xx = x[i]
    yy = y[i]
    zz = z[i]
    if grouplabel[i] != -1:
      if volume[xx, yy, zz] != -1 and volume_seg[xx, yy, zz] == groupseg[grouplabel[i]]:
        overlapgroupcounts[grouplabel[i], volume[xx, yy, zz]] += 1
    groupcounts[grouplabel[i]] += 1

  groupcate = np.argmax(overlapgroupcounts, axis=1)
  maxoverlapgroupcounts = np.max(overlapgroupcounts, axis=1)
  curr_max = np.max(volume)
  for i in range(groupcate.shape[0]):
    if maxoverlapgroupcounts[i] < 7 and groupcounts[i] > 12:
      curr_max += 1
      groupcate[i] = curr_max

  finalgrouplabel = -1 * np.ones(pts.shape[0])
  for i in range(pts.shape[0]):
    if grouplabel[i] != -1 and volume[x[i], y[i], z[i]] == -1:
      volume[x[i], y[i], z[i]] = groupcate[grouplabel[i]]
      volume_seg[x[i], y[i], z[i]] = groupseg[grouplabel[i]]
      finalgrouplabel[i] = groupcate[grouplabel[i]]
  return finalgrouplabel


def evaluation():

  ap_25 = DetectionAPCalculator(0.25)
  ap_50 = DetectionAPCalculator(0.50)

  mean_insSize_by_sem = get_mean_insSize_by_sem(DATASET_PATH, TRAIN_AREA)

  res_scenes = sorted(os.listdir(RESULT_PATH + 'res_by_scene/'))
  for scene_name in tqdm.tqdm(res_scenes):
    scene_result = scipy.io.loadmat(
        RESULT_PATH + 'res_by_scene/' + scene_name, verify_compressed_data_integrity=False)

    pc_all = []
    ins_gt_all = []
    sem_pred_all = []
    sem_gt_all = []
    gap = 5e-3
    volume_num = int(1. / gap) + 2
    volume = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)
    volume_sem = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)

    # Block merge predictions.
    bboxes_ptc_pred = []
    for i in range(len(scene_result)):
      block = 'block_' + str(i).zfill(4)
      if block not in scene_result:
        continue
      pc = scene_result[block][0]['pc'][0]
      ins_gt = scene_result[block][0]['ins_gt'][0][0]
      sem_gt = scene_result[block][0]['sem_gt'][0][0]
      bbscore_pred_raw = scene_result[block][0]['bbscore_pred_raw'][0][0]
      pmask_pred_raw = scene_result[block][0]['pmask_pred_raw'][0]
      sem_pred_raw = scene_result[block][0]['sem_pred_raw'][0]
      bboxes_ptc_pred.append(np.hstack((scene_result[block][0]['bbvert_pred_raw'][0].reshape(-1, 6),
                             scene_result[block][0]['bbscore_pred_raw'][0][0][:, None])))

      sem_pred = np.argmax(sem_pred_raw, axis=-1)
      pmask_pred = pmask_pred_raw * np.tile(bbscore_pred_raw[:, None],
                                            [1, pmask_pred_raw.shape[-1]])
      ins_pred = np.argmax(pmask_pred, axis=-2)
      ins_sem_dic = get_sem_for_ins(ins_by_pts=ins_pred, sem_by_pts=sem_pred)
      block_merge(volume, volume_sem, pc[:, 6:9], ins_pred, ins_sem_dic, gap)

      pc_all.append(pc)
      ins_gt_all.append(ins_gt)
      sem_pred_all.append(sem_pred)
      sem_gt_all.append(sem_gt)

    pc_all = np.concatenate(pc_all, axis=0)
    ins_gt_all = np.concatenate(ins_gt_all, axis=0)
    sem_pred_all = np.concatenate(sem_pred_all, axis=0)
    sem_gt_all = np.concatenate(sem_gt_all, axis=0)

    pc_xyz_int = (pc_all[:, 6:9] / gap).astype(np.int32)
    ins_pred_all = volume[tuple(pc_xyz_int.T)]
    pc_xyz_f = pc_xyz_int.astype(float)

    # Get bounding box from groundtruth.
    inst_gt_mask = np.logical_and(7 <= sem_gt_all, sem_gt_all <= 11)
    bboxes_gt, _ = get_bbox(pc_xyz_f, sem_gt_all, ins_gt_all, inst_gt_mask, -1)

    # Get bounding box from prediction.
    ins2sem = get_sem_for_ins(ins_pred_all, sem_pred_all)
    inst_pred_mask = np.full(ins_pred_all.shape[0], False)
    for ins_id, sem_cls in ins2sem.items():
      # Reject small instances (as in the evaluator of bonet)
      ins_mask = ins_pred_all == ins_id
      if ins_mask.sum() <= 0.2 * mean_insSize_by_sem[sem_cls]:
        continue
      # Reject non-movable objects.
      inst_pred_mask[ins_mask] = 7 <= sem_cls <= 11
    bboxes_pred, _ = get_bbox(pc_xyz_f, sem_pred_all, ins_pred_all, inst_pred_mask, -1,
                              strict_semantic_match=False)
    bboxes_ptc_pred = np.vstack(bboxes_ptc_pred)
    # Retrieve score from highest overlapping bounding box.
    overlaps = compute_overlaps(bboxes_ptc_pred[:, :6] / gap, bboxes_pred[:, :6]).numpy()
    bboxes_pred_score = bboxes_ptc_pred[overlaps.argmax(0), 6]

    # Run per-class nms
    bboxes_pred = np.hstack((bboxes_pred, bboxes_pred_score[:, None]))
    bboxes_pred_nms = []
    for i in np.unique(bboxes_pred[:, 6]):
      class_mask = bboxes_pred[:, 6] == i
      class_bboxes_pred = torch.from_numpy(bboxes_pred[class_mask]).float().to('cuda')
      nms_boxes, _, nms_scores = non_maximum_suppression(
          class_bboxes_pred[:, :6], None, class_bboxes_pred[:, 7], NMS_THRES, 1000, False, True)
      bboxes_pred_nms.append(np.hstack((nms_boxes.cpu().numpy(),
                                        np.ones((nms_boxes.shape[0], 1)) * i,
                                        nms_scores.cpu().numpy()[:, None])))
    if bboxes_pred_nms:
      bboxes_pred_nms = np.vstack(bboxes_pred_nms)
    else:
      bboxes_pred_nms = np.zeros((0, 8))

    # Format prediction and ground-truth to fit evaluator.
    gt_boxes = [[
        (int(bbox_cls - 7), bbox)
        for bbox, bbox_cls in zip(bboxes2corners(bboxes_gt[:, :6], swap_yz=True), bboxes_gt[:, 6])]
        if bboxes_gt.size != 0 else []
    ]
    pred_boxes = [[
        (int(bbox_cls - 7), bbox - 1e-6, bbox_score)
        for bbox, bbox_cls, bbox_score in zip(bboxes2corners(bboxes_pred_nms[:, :6], swap_yz=True),
                                              bboxes_pred_nms[:, 6], bboxes_pred_nms[:, 7])]
        if bboxes_pred_nms.size != 0 else []
    ]
    ap_25.step(pred_boxes, gt_boxes)
    ap_50.step(pred_boxes, gt_boxes)

  # Evaluate AP for all scenes.
  print('Finished parsing predictions. Start evaluating mAP.')
  print(ap_25.compute_metrics())
  print(ap_50.compute_metrics())


if __name__ == '__main__':
  evaluation()
