# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import warnings
import numpy as np
from scipy.spatial import ConvexHull
from multiprocessing import Pool


def box3d_vol(corners):
  ''' corners: (8,3) no assumption on axis direction '''
  a = np.sqrt(np.sum((corners[0, :] - corners[1, :])**2))
  b = np.sqrt(np.sum((corners[1, :] - corners[2, :])**2))
  c = np.sqrt(np.sum((corners[0, :] - corners[4, :])**2))
  return a * b * c


def polygon_clip(subjectPolygon, clipPolygon):
  """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """

  def inside(p):
    return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

  def computeIntersection():
    dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
    dp = [s[0] - e[0], s[1] - e[1]]
    n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    n2 = s[0] * e[1] - s[1] * e[0]
    n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
    return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

  outputList = subjectPolygon
  cp1 = clipPolygon[-1]

  for clipVertex in clipPolygon:
    cp2 = clipVertex
    inputList = outputList
    outputList = []
    s = inputList[-1]

    for subjectVertex in inputList:
      e = subjectVertex
      if inside(e):
        if not inside(s):
          outputList.append(computeIntersection())
        outputList.append(e)
      elif inside(s):
        outputList.append(computeIntersection())
      s = e
    cp1 = cp2
    if len(outputList) == 0:
      return None
  return (outputList)


def poly_area(x, y):
  """ Ref:
    http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
  return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convex_hull_intersection(p1, p2):
  """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
  inter_p = polygon_clip(p1, p2)
  if inter_p is not None:
    hull_inter = ConvexHull(inter_p)
    return inter_p, hull_inter.volume
  else:
    return None, 0.0


def box3d_iou(corners1, corners2):
  ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''
  # corner points are in counter clockwise order
  rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
  rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
  area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
  area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
  inter, inter_area = convex_hull_intersection(rect1, rect2)
  iou_2d = inter_area / (area1 + area2 - inter_area)
  ymax = min(corners1[0, 1], corners2[0, 1])
  ymin = max(corners1[4, 1], corners2[4, 1])
  inter_vol = inter_area * max(0.0, ymax - ymin)
  vol1 = box3d_vol(corners1)
  vol2 = box3d_vol(corners2)
  iou = inter_vol / (vol1 + vol2 - inter_vol)
  return iou, iou_2d


def get_iou_obb(bb1, bb2):
  iou3d, iou2d = box3d_iou(bb1, bb2)
  return iou3d


def get_iou(box_a, box_b):
  """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths
    Returns:
        iou
    """

  max_a = box_a[0:3] + box_a[3:6] / 2
  max_b = box_b[0:3] + box_b[3:6] / 2
  min_max = np.array([max_a, max_b]).min(0)

  min_a = box_a[0:3] - box_a[3:6] / 2
  min_b = box_b[0:3] - box_b[3:6] / 2
  max_min = np.array([min_a, min_b]).max(0)
  if not ((min_max > max_min).all()):
    return 0.0

  intersection = (min_max - max_min).prod()
  vol_a = box_a[3:6].prod()
  vol_b = box_b[3:6].prod()
  union = vol_a + vol_b - intersection
  return 1.0 * intersection / union


def get_iou_main(get_iou_func, args):
  return get_iou_func(*args)


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def eval_det_cls(pred, gt, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
  """ Generic functions to compute precision/recall for object detection
        for a single class.
        Input:
            pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
            gt: map of {img_id: [bbox]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if True use VOC07 11 point method
        Output:
            rec: numpy array of length nd
            prec: numpy array of length nd
            ap: scalar, average precision
    """

  # construct gt objects
  class_recs = {}  # {img_id: {'bbox': bbox list, 'det': matched list}}
  npos = 0
  for img_id in gt.keys():
    bbox = np.array(gt[img_id])
    det = [False] * len(bbox)
    npos += len(bbox)
    class_recs[img_id] = {'bbox': bbox, 'det': det}
  # pad empty list to all other imgids
  for img_id in pred.keys():
    if img_id not in gt:
      class_recs[img_id] = {'bbox': np.array([]), 'det': []}

  # construct dets
  image_ids = []
  confidence = []
  BB = []
  for img_id in pred.keys():
    for box, score in pred[img_id]:
      image_ids.append(img_id)
      confidence.append(score)
      BB.append(box)
  confidence = np.array(confidence)
  BB = np.array(BB)  # (nd,4 or 8,3 or 6)

  # sort by confidence
  sorted_ind = np.argsort(-confidence)
  BB = BB[sorted_ind, ...]
  image_ids = [image_ids[x] for x in sorted_ind]

  # go down dets and mark TPs and FPs
  nd = len(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)
  for d in range(nd):
    R = class_recs[image_ids[d]]
    bb = BB[d, ...].astype(float)
    ovmax = -np.inf
    BBGT = R['bbox'].astype(float)

    if BBGT.size > 0:
      # compute overlaps
      for j in range(BBGT.shape[0]):
        iou = get_iou_main(get_iou_func, (bb, BBGT[j, ...]))
        if iou > ovmax:
          ovmax = iou
          jmax = j

    if ovmax > ovthresh:
      if not R['det'][jmax]:
        tp[d] = 1.
        R['det'][jmax] = 1
      else:
        fp[d] = 1.
    else:
      fp[d] = 1.

  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)

  return rec, prec, ap


def eval_det_cls_wrapper(arguments):
  pred, gt, ovthresh, use_07_metric, get_iou_func = arguments
  rec, prec, ap = eval_det_cls(pred, gt, ovthresh, use_07_metric, get_iou_func)
  return (rec, prec, ap)


def eval_det_multiprocessing(pred_all,
                             gt_all,
                             ovthresh=0.25,
                             use_07_metric=False,
                             get_iou_func=get_iou):
  """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
  pred = {}  # map {classname: pred}
  gt = {}  # map {classname: gt}
  for img_id in pred_all.keys():
    for classname, bbox, score in pred_all[img_id]:
      if classname not in pred:
        pred[classname] = {}
      if img_id not in pred[classname]:
        pred[classname][img_id] = []
      if classname not in gt:
        gt[classname] = {}
      if img_id not in gt[classname]:
        gt[classname][img_id] = []
      pred[classname][img_id].append((bbox, score))
  for img_id in gt_all.keys():
    for classname, bbox in gt_all[img_id]:
      if classname not in gt:
        gt[classname] = {}
      if img_id not in gt[classname]:
        gt[classname][img_id] = []
      gt[classname][img_id].append(bbox)

  rec = {}
  prec = {}
  ap = {}
  p = Pool(processes=10)
  ret_values = p.map(eval_det_cls_wrapper,
                     [(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func)
                      for classname in gt.keys()
                      if classname in pred])
  p.close()
  for i, classname in enumerate(gt.keys()):
    if classname in pred:
      rec[classname], prec[classname], ap[classname] = ret_values[i]
    else:
      rec[classname] = 0
      prec[classname] = 0
      ap[classname] = 0

  return rec, prec, ap


class DetectionAPCalculator:
  '''Calculating Average Precision'''

  def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
    """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
    self.ap_iou_thresh = ap_iou_thresh
    self.class2type_map = class2type_map
    self.reset()

  def step(self, batch_pred_map_cls, batch_gt_map_cls):
    """ Accumulate one batch of prediction and groundtruth.

        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """

    bsize = len(batch_pred_map_cls)
    assert (bsize == len(batch_gt_map_cls))
    for i in range(bsize):
      self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
      self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
      self.scan_cnt += 1

  def compute_metrics(self):
    """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
    rec, prec, ap = eval_det_multiprocessing(
        self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh, get_iou_func=get_iou_obb)
    ret_dict = {'rec': rec, 'prec': prec}
    for key in sorted(ap.keys()):
      clsname = self.class2type_map[key] if self.class2type_map else str(key)
      ret_dict['%s Average Precision' % (clsname)] = ap[key]
    ret_dict['mAP'] = np.mean(list(ap.values()))
    rec_list = []
    for key in sorted(ap.keys()):
      clsname = self.class2type_map[key] if self.class2type_map else str(key)
      try:
        ret_dict['%s Recall' % (clsname)] = rec[key][-1]
        rec_list.append(rec[key][-1])
      except:  # yapf: disable
        ret_dict['%s Recall' % (clsname)] = 0
        rec_list.append(0)
    ret_dict['AR'] = np.mean(rec_list)
    return ret_dict

  def reset(self):
    self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
    self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
    self.scan_cnt = 0
