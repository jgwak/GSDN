import numpy as np
import torch

import MinkowskiEngine as ME
import detectron3d._C as custom


def sparse2dense(sparse_tensor, pixel_dist, voxel_shape):
  dense_tensor, min_coords, tensor_stride = sparse_tensor.dense(
      min_coords=torch.zeros(3).int(),
      max_coords=voxel_shape - pixel_dist)  # max_coords inclusive
  return dense_tensor


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             get_negative_anchors=False):
  anchors = []
  anchor_coords = []
  ratios = np.sqrt(ratios)
  for scale, feature_shape, feature_stride in zip(scales, feature_shapes, feature_strides):
    # Get all combinations of scale and ratios
    scale, ratio_idxs = np.meshgrid(np.array(scale), np.arange(ratios.shape[0]))
    scale = scale.flatten()
    ratios = ratios[ratio_idxs.flatten()]

    # Enumerate size from scale and ratios
    size_x = scale / ratios[:, 0]
    size_y = scale / ratios[:, 1]
    size_z = scale / ratios[:, 2]

    # Enumerate shifts in feature space
    base = 0
    if get_negative_anchors:
      base = -int(2 ** (np.log2(feature_strides[-1]) - np.log2(feature_stride)))
    shifts_x = np.arange(base, feature_shape[0]) * feature_stride + 0.5 * feature_stride
    shifts_y = np.arange(base, feature_shape[1]) * feature_stride + 0.5 * feature_stride
    shifts_z = np.arange(base, feature_shape[2]) * feature_stride + 0.5 * feature_stride
    shifts_x, shifts_y, shifts_z = np.meshgrid(shifts_x, shifts_y, shifts_z, indexing='ij')

    # Enumerate combinations of shifts and size
    box_size_x, box_centers_x = np.meshgrid(size_x, shifts_x)
    box_size_y, box_centers_y = np.meshgrid(size_y, shifts_y)
    box_size_z, box_centers_z = np.meshgrid(size_z, shifts_z)

    # Reshape to get a list of (x, x, z) and a list of (h, w, l)
    box_centers = np.stack([box_centers_x, box_centers_y, box_centers_z], axis=2).reshape(-1, 3)
    box_sizes = np.stack([box_size_x, box_size_y, box_size_z], axis=2).reshape([-1, 3])

    # Convert to corner coordinates (x1, y1, z1, x2, y2, z2)
    anchors.append(
        np.concatenate([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1))
    anchor_coords.append(box_centers.astype(int))
  anchors = np.concatenate(anchors, axis=0).astype(np.float32)
  return anchors, anchor_coords


def get_bbox_target(pred_box, gt_box, delta_std):
  assert pred_box.shape == gt_box.shape
  box_dim = pred_box.shape[1] // 2
  pred_size = pred_box[:, box_dim:] - pred_box[:, :box_dim]
  pred_center = pred_box[:, :box_dim] + 0.5 * pred_size
  gt_size = gt_box[:, box_dim:] - gt_box[:, :box_dim]
  gt_center = gt_box[:, :box_dim] + 0.5 * gt_size
  delta_center = (gt_center - pred_center) / pred_size
  delta_size = torch.log(gt_size / pred_size)
  delta = torch.cat((delta_center, delta_size), 1)
  return delta / torch.from_numpy(np.array(delta_std)).to(delta)


def compute_axisaligned_overlaps(boxes1, boxes2):
  boxes1_np = boxes1.cpu().numpy()
  boxes2_np = boxes2.cpu().numpy()
  box_dim = boxes1_np.shape[1] // 2
  area1 = np.prod(boxes1_np[:, 3:] - boxes1_np[:, :3], 1)
  area2 = np.prod(boxes2_np[:, 3:] - boxes2_np[:, :3], 1)
  ious = np.zeros((boxes1_np.shape[0], boxes2_np.shape[0]), dtype=np.float32)
  for i, (box2, box2area) in enumerate(zip(boxes2_np, area2)):
    min_max = np.minimum(boxes1_np[:, box_dim:], np.expand_dims(box2[box_dim:], 0))
    max_min = np.maximum(boxes1_np[:, :box_dim], np.expand_dims(box2[:box_dim], 0))
    intersection = np.prod(np.maximum(min_max - max_min, 0), 1)
    union = area1 + box2area - intersection + 1e-7
    ious[:, i] = intersection / union
  return torch.from_numpy(ious).to(boxes1)


def compute_overlaps(boxes1, boxes2, rotations1=None, rotations2=None):
  if isinstance(boxes1, np.ndarray) and isinstance(boxes2, np.ndarray):
    boxes1 = torch.from_numpy(boxes1).float()
    boxes2 = torch.from_numpy(boxes2).float()
  if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
    return torch.zeros((boxes1.shape[0], boxes2.shape[0])).to(boxes1)
  if rotations1 is not None:
    if isinstance(rotations1, np.ndarray):
      rotations1 = torch.from_numpy(rotations1).float()
    boxes1 = get_axisaligned_bbox(boxes1, rotations1)
  if rotations2 is not None:
    if isinstance(rotations2, np.ndarray):
      rotations2 = torch.from_numpy(rotations2).float()
    boxes2 = get_axisaligned_bbox(boxes2, rotations2)
  return compute_axisaligned_overlaps(boxes1, boxes2)


def _get_box_normalization_param(max_ptc_size):
  shift = np.concatenate((np.zeros_like(max_ptc_size), np.ones_like(max_ptc_size)))
  scale = np.concatenate((max_ptc_size, max_ptc_size)) - 1
  return shift, scale


def map_coordinates(in_sinput, out_coords, force_stride=False, check_input_map=False,
                    check_output_map=False):
  if out_coords.shape[0] == 0:
    return [torch.zeros(0, dtype=torch.long)], [torch.zeros(0, dtype=torch.long)]
  cm = in_sinput.coords_man
  out_key = cm.create_coords_key(out_coords)
  if force_stride:
    out_key = cm.stride(out_key, in_sinput.tensor_stride[0], force_creation=True)
  in2out, out2in = cm.get_kernel_map(in_sinput.coords_key, out_key, kernel_size=1, region_type=1)
  if check_input_map and out2in[0].shape[0] != in_sinput.coords.shape[0]:
    raise ValueError('Found missing input coordinates in the coordinate map')
  if check_output_map and in2out[0].shape[0] != out_coords.shape[0]:
    raise ValueError('Found missing output coordinates in the coordinate map')
  return in2out, out2in


def normalize_boxes(boxes, max_ptc_size):
  def _normalize_boxes(boxes, max_ptc_size):
    num_bbox_params = len(max_ptc_size * 2)
    assert boxes.shape[1] % num_bbox_params == 0
    original_box_shape = boxes.shape
    shift, scale = _get_box_normalization_param(max_ptc_size)
    boxes = boxes.reshape(-1, num_bbox_params)
    boxes = np.divide((boxes - shift), scale).astype(np.float32)
    return boxes.reshape(*original_box_shape)
  if isinstance(boxes, ME.SparseTensor):
    normalized_boxes = ME.SparseTensor(
        torch.from_numpy(_normalize_boxes(boxes.F.numpy(), max_ptc_size)),
        coords_key=boxes.coords_key, coords_manager=boxes.coords_man)
  elif isinstance(boxes, torch.Tensor):
    normalized_boxes = torch.from_numpy(
        _normalize_boxes(boxes.cpu().numpy(), max_ptc_size)).to(boxes)
  else:
    normalized_boxes = _normalize_boxes(boxes, max_ptc_size)
  return normalized_boxes


def unnormalize_boxes(boxes, max_ptc_size):
  assert boxes.shape[1] == len(max_ptc_size) * 2
  shift, scale = _get_box_normalization_param(max_ptc_size)
  return np.multiply(boxes, scale) + shift


def apply_box_deltas(boxes, deltas, normalize_bbox):
  assert deltas.shape == boxes.shape
  box_dim = deltas.shape[-1] // 2
  box_size = boxes[..., box_dim:] - boxes[..., :box_dim]
  box_center = boxes[..., :box_dim] + 0.5 * box_size
  box_center += deltas[..., :box_dim] * box_size
  box_size *= torch.exp(deltas[..., box_dim:])
  box_size = torch.clamp(box_size, 0., 1.)
  if normalize_bbox:
    box_center = torch.clamp(box_center, 0., 1.)
  bbox = torch.cat((box_center - 0.5 * box_size, box_center + 0.5 * box_size), -1)
  if normalize_bbox:
    return torch.clamp(bbox, 0., 1.)
  return bbox


def check_backbone_shapes(backbone_shapes, fpn_outputs):
  fpn_output_shapes = np.array([o.shape[2:] for o in fpn_outputs])
  assert np.allclose(backbone_shapes, fpn_output_shapes), 'FPN output shape mismatch'


def normalize_rotation(rotation):
  return torch.atan2(torch.sin(rotation), torch.cos(rotation))


def apply_rotations(coords, rotations):
  rot_s, rot_c = torch.sin(rotations), torch.cos(rotations)
  rotations_m = torch.eye(3)[None, :, :].repeat(coords.shape[0], 1, 1).to(coords)
  rot_z = torch.stack((torch.stack((rot_c, -rot_s)), torch.stack((rot_s, rot_c))))
  rotations_m[:, :2, :2] = rot_z.permute(2, 0, 1).to(coords)
  return torch.bmm(rotations_m, coords)


def get_axisaligned_bbox(boxes, rotations):
  if boxes.shape[0] == 0:
    return boxes
  boxes_size = boxes[:, 3:] - boxes[:, :3]
  boxes_center = (boxes[:, 3:] + boxes[:, :3]) / 2
  x, y, z = boxes_size[:, 0], boxes_size[:, 1], boxes_size[:, 2]
  corners = torch.stack([torch.stack([x, x, -x, -x, x, x, -x, -x]),
                         torch.stack([y, -y, -y, y, y, -y, -y, y]),
                         torch.stack([z, z, z, z, -z, -z, -z, -z])]) / 2
  corners = corners.permute(2, 0, 1).to(boxes)
  corners_rotated = apply_rotations(corners, rotations)
  boxes_rotated = torch.cat((corners_rotated.min(2)[0], corners_rotated.max(2)[0]), 1)
  return boxes_rotated + boxes_center.repeat(1, 2)


def non_maximum_suppression(boxes, rotations, scores, nms_threshold, proposal_count,
                            detection_rot_nms, aggregate_overlap, return_scores=None,
                            return_rotations=None):
  assert boxes.shape[0] == scores.shape[0]
  return_scores = (return_scores if return_scores is not None else scores).clone()
  return_rotations = return_rotations if return_rotations is not None else rotations
  scores += 1e-4
  boxes_nms = boxes
  if rotations is not None and detection_rot_nms:
    boxes_nms = get_axisaligned_bbox(boxes, rotations)
  picks = custom.nms(boxes_nms, scores, nms_threshold)
  unique_picks = torch.unique(picks)
  num_picks = min(len(unique_picks), proposal_count)
  nms_picks = unique_picks[:num_picks]
  if aggregate_overlap:
    scores_ = scores.unsqueeze(1)
    boxes1 = torch.cat((boxes * scores_, scores_), 1)
    boxes_sum = torch.zeros(boxes.shape[0], boxes.shape[1] + 1).to(boxes)
    boxes_sum.index_add_(0, picks, boxes1)
    unique_boxes_sum = boxes_sum[nms_picks]
    nms_boxes = unique_boxes_sum[:, :6] / unique_boxes_sum[:, 6, None]
    nms_scores = return_scores[nms_picks]
    nms_rotations = None
    if return_rotations is not None:
      rotations1 = torch.stack((torch.sin(return_rotations) * scores,
                                torch.cos(return_rotations) * scores, scores)).T
      rotations_sum = torch.zeros(return_rotations.shape[0], 3).to(return_rotations)
      rotations_sum.index_add_(0, picks, rotations1)
      rotations_sincos = (rotations_sum[:, :2] / rotations_sum[:, 2, None])[nms_picks]
      nms_rotations = torch.atan2(rotations_sincos[:, 0], rotations_sincos[:, 1])
  else:
    nms_boxes = boxes[nms_picks]
    nms_rotations = None if return_rotations is None else return_rotations[nms_picks]
    nms_scores = return_scores[nms_picks]
  return nms_boxes, nms_rotations, nms_scores
