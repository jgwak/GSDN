import numpy as np
import matplotlib.pyplot as plt

import lib.detection_utils as detection_utils


TRAIN_BBOXES = 'scannet_train_gtbboxes.npy'  # npy array of (num_bboxes, 6)

RPN_ANCHOR_SCALE_BASES = (2, 3, 4, 5, 6, 7, 8)
RPN_NUM_SCALES = 4
ANCHOR_RATIOS = (1.5, 2, 3, 4, 5)

FPN_BASE_LEVEL = 3
FPN_MAX_SCALES = (32, 40, 48, 56, 64)


def search_anchor_params(train_bboxes):
  for scale_base in RPN_ANCHOR_SCALE_BASES:
    for anchor_ratio in ANCHOR_RATIOS:
      sanchor_ratio = np.sqrt(anchor_ratio)
      scales = [scale_base * 2 ** i for i in range(RPN_NUM_SCALES)]
      ratios = np.array([[1 / sanchor_ratio, 1 / sanchor_ratio, sanchor_ratio],
                         [1 / sanchor_ratio, sanchor_ratio, 1 / sanchor_ratio],
                         [sanchor_ratio, 1 / sanchor_ratio, 1 / sanchor_ratio],
                         [sanchor_ratio, sanchor_ratio, 1 / sanchor_ratio],
                         [sanchor_ratio, 1 / sanchor_ratio, sanchor_ratio],
                         [1 / sanchor_ratio, sanchor_ratio, sanchor_ratio],
                         [1, 1, 1]])
      anchors = np.vstack([ratios * scale for scale in scales])
      targets = train_bboxes[:, 3:] - train_bboxes[:, :3]
      anchors_bboxes = np.hstack((-anchors / 2, anchors / 2))
      targets_bboxes = np.hstack((-targets / 2, targets / 2))
      overlaps = detection_utils.compute_overlaps(anchors_bboxes, targets_bboxes).max(0)[0]
      plt.hist(overlaps, range=(0, 1))
      axes = plt.gca()
      axes.set_ylim([0, 1400])
      plt.savefig(f'anchor_scale{scale_base}_ratio{anchor_ratio}.png')
      plt.clf()
      print(f'scale: {scale_base}, ratio: {anchor_ratio}:\tmin: {overlaps.min().item():.4f}'
            f'\tavg: {overlaps.mean().item():.4f}')


def search_fpn_params(train_bboxes):
  for fpn_max_scale in FPN_MAX_SCALES:
    target_scale = np.prod(train_bboxes[:, 3:] - train_bboxes[:, :3], 1)
    target_level = FPN_BASE_LEVEL + np.log2(np.cbrt(target_scale) / fpn_max_scale)
    plt.hist(target_level, range(-2, 5))
    plt.savefig(f'fpn_scale{fpn_max_scale}.png')
    plt.clf()


if __name__ == '__main__':
  train_bboxes = np.load(TRAIN_BBOXES)
  search_anchor_params(train_bboxes)
  search_fpn_params(train_bboxes)
