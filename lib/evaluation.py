import numpy as np
from lib.utils import compute_iou_3d


def get_single_mAP(precision, iou_thrs, output_iou_thr):
  """
  Parameters:
    precision:  T x R X K matrix of precisions from accumulate()
    iouThrs: IoU thresholds
    output_iou_thr: iou threshold for mAP. If None, we calculate for range 0.5 : 0.95
  """
  s = precision
  if output_iou_thr:
    t = np.where(output_iou_thr == iou_thrs)[0]
    s = precision[t]
  if len(s[s > -1]) == 0:
    mean_s = -1
  else:
    mean_s = np.mean(s[s > -1])
  return mean_s


def accumulate(detection_matches, class_ids, num_gt_boxes, iou_thrs, rec_thrs):
  """
  Parameters:
    detection_matches: list of dtm arrays (from match_dt_2_gt()) of length nClasses
    class_ids: list of class Ids
    num_gt_boxes: list of number of gt boxes per class
    iou_thrs, rec_thrs: iou and recall thresholds

  Returns:
    precision: T x R X K matrix of precisions over IoU thresholds, recall thresholds, and classes
  """
  T = len(iou_thrs)
  R = len(rec_thrs)
  K = len(class_ids)
  precision = -1 * np.ones((T, R, K))  # -1 for the precision of absent categories
  for k, category in enumerate(class_ids):
    dtm = detection_matches[k]
    if dtm is None:
      continue
    D = dtm.shape[1]
    G = num_gt_boxes[k]
    if G == 0:
      continue
    tps = (dtm != -1)  # get all matched detections mask
    fps = (dtm == -1)
    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
      tp = np.array(tp)
      fp = np.array(fp)
      rc = tp / G
      pr = tp / (fp + tp + np.spacing(1))
      q = np.zeros((R,))  # stores precisions for each recall value

      # use python array gets significant speed improvement
      pr = pr.tolist()
      for i in range(D - 1, 0, -1):
        if pr[i] > pr[i - 1]:
          pr[i - 1] = pr[i]

      inds = np.searchsorted(rc, rec_thrs, side='left')
      try:
        for ri, pi in enumerate(inds):
          q[ri] = pr[pi]
      except IndexError:
        pass
      precision[t, :, k] = np.array(q)
  return precision


def match_dt_2_gt(ious, iou_thrs):
  """ Matches detection bboxes to groundtruth
  Parameters:
    ious: D x G matrix, where D = nDetections and G = nGroundTruth bboxes.
        Each element stores the iou between a detection and a groundtruth box.
        Note that detections are first sorted by decreasing score before calculating ious
    iou_thrs: T X 1 array of iou thresholds to perform matching
  Returns:
    dtm: T x D array storing the index of each GT match for each thresh, or -1 if no match
  """
  T = len(iou_thrs)
  D, G = np.shape(ious)
  if D == 0 and G == 0:  # no boxes in gt or dt
    return None
  gtm = -1 * np.ones((T, G))
  dtm = -1 * np.ones((T, D))
  for tind, t in enumerate(iou_thrs):
    for dind in range(D):
      # information about best match so far (m=-1 -> unmatched)
      iou = min([t, 1 - 1e-10])
      m = -1
      for gind in range(G):
        # if this gind already matched, continue
        if gtm[tind, gind] > -1:
          continue
        # continue to next gt unless better match made
        if ious[dind, gind] < iou:
          continue
        # if match successful and best so far, store appropriately
        iou = ious[dind, gind]
        m = gind
      # if match made store id of match for both dt and gt
      if m == -1:
        continue
      dtm[tind, dind] = m
      gtm[tind, m] = dind
  return dtm


def get_mAP_scores(dt_boxes, gt_boxes, class_ids):
  """ Calculates mAP scores for mAP for IoU 0.5, 0.75, and 0.5 : 0.95
  Parameters:
    dt_boxes: D x 8 matrix of detection boxes, representing (x, y, z, w, l, h, class, score)
    gt_boxes: D x 7 matrix of grondtruth boxes, representing (x, y, z, w, l, h, class)
    class_ids: list of class Ids
  Returns:
    mAP: tuple of length 3 for mAP with IoU threshold (0.5, 0.75, 0.5:0.95)
  """
  output_iou_thrs = [0.25, 0.5]
  iou_thrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .05) + 1, endpoint=True)
  rec_thrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)

  dt_boxes = dt_boxes[dt_boxes[:, -1].argsort()[::-1]]  # sort detections by decreasing scores
  dt_classes = dt_boxes[:, -2]
  gt_classes = gt_boxes[:, -1]
  detection_matches = []  # matches for each class
  num_gt_boxes = []
  for k in class_ids:
    dt_boxes_k = dt_boxes[dt_classes == k]
    gt_boxes_k = gt_boxes[gt_classes == k]
    D = dt_boxes_k.shape[0]
    G = gt_boxes_k.shape[0]
    ious = np.zeros((D, G))  # precompute all IoUs
    for d in range(D):
      for g in range(G):
        ious[d, g] = compute_iou_3d(dt_boxes_k[d, :6], gt_boxes_k[g, :6])
    dtm = match_dt_2_gt(ious, iou_thrs)
    detection_matches.append(dtm)
    num_gt_boxes.append(G)
  precision = accumulate(detection_matches, class_ids, num_gt_boxes, iou_thrs, rec_thrs)

  mAP = []
  for output_iou_thr in output_iou_thrs:
    mAP.append(get_single_mAP(precision, iou_thrs, output_iou_thr))
  return mAP
