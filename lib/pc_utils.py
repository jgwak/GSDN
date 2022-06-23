import logging
import os
import os.path as osp

import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank, inv
from plyfile import PlyData, PlyElement
from retrying import retry

COLOR_MAP_RGB = (
    (174., 199., 232.),
    (152., 223., 138.),
    (31., 119., 180.),
    (255., 187., 120.),
    (188., 189., 34.),
    (140., 86., 75.),
    (255., 152., 150.),
    (214., 39., 40.),
    (197., 176., 213.),
    (148., 103., 189.),
    (196., 156., 148.),
    (23., 190., 207.),
    (247., 182., 210.),
    (66., 188., 102.),
    (219., 219., 141.),
    (140., 57., 197.),
    (202., 185., 52.),
    (51., 176., 203.),
    (200., 54., 131.),
    (92., 193., 61.),
    (78., 71., 183.),
    (172., 114., 82.),
    (255., 127., 14.),
    (91., 163., 138.),
    (153., 98., 156.),
    (140., 153., 101.),
    (158., 218., 229.),
    (100., 125., 154.),
    (178., 127., 135.),
    (146., 111., 194.),
    (44., 160., 44.),
    (112., 128., 144.),
    (96., 207., 209.),
    (227., 119., 194.),
    (213., 92., 176.),
    (94., 106., 211.),
    (82., 84., 163.),
    (100., 85., 144.),
)
IGNORE_COLOR = (0, 0, 0)


def crop_pointcloud(pointcloud, center, clip_size, axis, ratio):
  clip_coord = clip_size * 2 * ratio - clip_size + center
  return pointcloud[pointcloud[:, axis] >= clip_coord]


def sample_faces(vertices, faces, n_samples=10**4):
  r"""
  Samples point cloud on the surface of the model defined as vectices and
  faces. This function uses vectorized operations so fast at the cost of some
  memory.

  Parameters:
    vertices  - n x 3 matrix
    faces     - n x 3 matrix
    n_samples - positive integer

  Return:
    vertices - point cloud

  Reference :
    [1] Barycentric coordinate system

    \begin{align}
      P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
    \end{align}
  """
  vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                       vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
  face_areas = np.sqrt(np.sum(vec_cross**2, 1))
  face_areas = face_areas / np.sum(face_areas)

  # Sample exactly n_samples. First, oversample points and remove redundant
  # Contributed by Yangyan (yangyan.lee@gmail.com)
  n_samples_per_face = np.ceil(n_samples * face_areas).astype(int)
  floor_num = np.sum(n_samples_per_face) - n_samples
  if floor_num > 0:
    indices = np.where(n_samples_per_face > 0)[0]
    floor_indices = np.random.choice(indices, floor_num, replace=True)
    n_samples_per_face[floor_indices] -= 1

  n_samples = np.sum(n_samples_per_face)

  # Create a vector that contains the face indices
  sample_face_idx = np.zeros((n_samples,), dtype=int)
  acc = 0
  for face_idx, _n_sample in enumerate(n_samples_per_face):
    sample_face_idx[acc:acc + _n_sample] = face_idx
    acc += _n_sample

  r = np.random.rand(n_samples, 2)
  A = vertices[faces[sample_face_idx, 0], :]
  B = vertices[faces[sample_face_idx, 1], :]
  C = vertices[faces[sample_face_idx, 2], :]
  P = ((1 - np.sqrt(r[:, 0:1])) * A + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B
       + np.sqrt(r[:, 0:1]) * r[:, 1:] * C)
  return P


def read_offfile(filepath):

  def _read_lines(f, n, dtype):
    return np.array([list(map(dtype, f.readline().rstrip().split())) for _ in range(n)])

  with open(filepath) as f:
    header = f.readline().rstrip()
    assert header.startswith('OFF')
    if header[3:]:
      n_verts, n_faces, _ = map(int, header[3:].split())
    else:
      n_verts, n_faces, _ = map(int, f.readline().rstrip().split())
    verts = _read_lines(f, n_verts, float)
    faces = _read_lines(f, n_faces, int)[:, 1:]
    assert not f.read(), 'Expected EOF'
  return verts, faces


def retry_on_ioerror(exc):
  logging.warning("Retrying file load")
  return isinstance(exc, IOError)


@retry(
    retry_on_exception=retry_on_ioerror,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    stop_max_delay=30000)
def read_plyfile(filepath):
  """Read ply file and return it as numpy array. Returns None if emtpy."""
  with open(filepath, 'rb') as f:
    plydata = PlyData.read(f)
  if plydata.elements:
    return pd.DataFrame(plydata.elements[0].data).values


def save_point_cloud(points_3d, filename, binary=True, with_label=False, verbose=True):
  """Save an RGB point cloud as a PLY file.

  Args:
    points_3d: Nx6 matrix where points_3d[:, :3] are the XYZ coordinates and points_3d[:, 4:] are
        the RGB values. If Nx3 matrix, save all points with [128, 128, 128] (gray) color.
  """
  assert points_3d.ndim == 2
  if with_label:
    assert points_3d.shape[1] == 7 or points_3d.shape[1] == 8
    if points_3d.shape[1] == 7:
      python_types = (float, float, float, int, int, int, int)
      npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                   ('blue', 'u1'), ('label', 'u1')]
    else:
      python_types = (float, float, float, int, int, int, int, int)
      npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                   ('blue', 'u1'), ('label_class', 'u1'), ('label_instance', 'u2')]
  else:
    if points_3d.shape[1] == 3:
      gray_concat = np.tile(np.array([128], dtype=np.uint8), (points_3d.shape[0], 3))
      points_3d = np.hstack((points_3d, gray_concat))
    assert points_3d.shape[1] == 6
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1')]
  if binary is True:
    # Format into NumPy structured array
    vertices = []
    for row_idx in range(points_3d.shape[0]):
      cur_point = points_3d[row_idx]
      vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
    vertices_array = np.array(vertices, dtype=npy_types)
    el = PlyElement.describe(vertices_array, 'vertex')

    # Write
    PlyData([el]).write(filename)
  else:
    # PlyData([el], text=True).write(filename)
    with open(filename, 'w') as f:
      f.write('ply\n'
              'format ascii 1.0\n'
              'element vertex %d\n'
              'property float x\n'
              'property float y\n'
              'property float z\n'
              'property uchar red\n'
              'property uchar green\n'
              'property uchar blue\n'
              'property uchar alpha\n'
              'end_header\n' % points_3d.shape[0])
      for row_idx in range(points_3d.shape[0]):
        X, Y, Z, R, G, B = points_3d[row_idx]
        f.write('%f %f %f %d %d %d 0\n' % (X, Y, Z, R, G, B))
  if verbose is True:
    print('Saved point cloud to: %s' % filename)


def quat_rotation(fr, to):
  h = fr + to
  h /= np.sqrt(np.sum(h**2))
  return np.array([
      np.dot(fr, h), fr[1] * h[2] - fr[2] * h[1], fr[2] * h[0] - fr[0] * h[2],
      fr[0] * h[1] - fr[1] * h[0]
  ])


def normal2quat(normals):
  # Use fr = [1, 0, 0], to = normals
  normals[:, 0] += 1
  normals /= np.sqrt(np.sum(normals**2, 1))
  return np.hstack((normals[:, 0], -normals[:, 2], normals[:, 1]))


def translate_point_cloud(points_3d, translation):
  """Translate points using the specified translation vector.

  Args:
    points_3d: Nx3 or Nx6 matrix
    translation: Vector of shape (3,) or (3, 1)
  """
  orig_points_3d = points_3d
  assert (orig_points_3d.shape[1] == 3) or (orig_points_3d.shape[1] == 6)
  if orig_points_3d.shape[1] == 6:  # XYZRGB
    points_3d = points_3d[:, :3]
  if translation.shape == (3,):
    translation = translation.reshape(3, 1)
  translated_points_3d = (points_3d.T + translation).T

  # Add color if appropriate
  if orig_points_3d.shape[1] == 6:  # XYZRGB
    translated_points_3d = np.hstack((translated_points_3d, orig_points_3d[:, -3:]))
  return translated_points_3d


class Camera(object):

  def __init__(self, intrinsics):
    self._intrinsics = intrinsics
    self._camera_matrix = self.build_camera_matrix(self.intrinsics)
    self._K_inv = inv(self.camera_matrix)

  @staticmethod
  def build_camera_matrix(intrinsics):
    """Build the 3x3 camera matrix K using the given intrinsics.

    Equation 6.10 from HZ.
    """
    f = intrinsics['focal_length']
    pp_x = intrinsics['pp_x']
    pp_y = intrinsics['pp_y']

    K = np.array([[f, 0, pp_x], [0, f, pp_y], [0, 0, 1]], dtype=np.float32)
    # K[:, 0] *= -1.  # Step 1 of Kyle
    assert matrix_rank(K) == 3
    return K

  @staticmethod
  def extrinsics2RT(extrinsics):
    """Convert extrinsics matrix to separate rotation matrix R and translation vector T.
    """
    assert extrinsics.shape == (4, 4)
    R = extrinsics[:3, :3]
    T = extrinsics[3, :3]
    R = np.copy(R)
    T = np.copy(T)
    T = T.reshape(3, 1)
    R[0, :] *= -1.  # Step 1 of Kyle
    T *= 100.  # Convert from m to cm
    return R, T

  def project(self, points_3d, extrinsics=None):
    """Project a 3D point in camera coordinates into the camera/image plane.

    Args:
      point_3d:
    """
    if extrinsics is not None:  # Map points to camera coordinates
      points_3d = self.world2camera(extrinsics, points_3d)

    raise NotImplementedError

  def backproject(self,
                  depth_map,
                  labels=None,
                  max_depth=None,
                  max_height=None,
                  min_height=None,
                  rgb_img=None,
                  extrinsics=None,
                  prune=True):
    """Backproject a depth map into 3D points (camera coordinate system). Attach color if RGB image
    is provided, otherwise use gray [128 128 128] color.

    Does not show points at Z = 0 or maximum Z = 65535 depth.

    Args:
      labels: Tensor with the same shape as depth map (but can be 1-channel or 3-channel).
      max_depth: Maximum depth in cm. All pts with depth greater than max_depth will be ignored.
      max_height: Maximum height in cm. All pts with height greater than max_height will be ignored.

    Returns:
      points_3d: Numpy array of size Nx3 (XYZ) or Nx6 (XYZRGB).
    """
    if labels is not None:
      assert depth_map.shape[:2] == labels.shape[:2]
      if (labels.ndim == 2) or ((labels.ndim == 3) and (labels.shape[2] == 1)):
        n_label_channels = 1
      elif (labels.ndim == 3) and (labels.shape[2] == 3):
        n_label_channels = 3

    if rgb_img is not None:
      assert depth_map.shape[:2] == rgb_img.shape[:2]
    else:
      rgb_img = np.ones_like(depth_map, dtype=np.uint8) * 255

    # Convert from 1-channel to 3-channel
    if (rgb_img.ndim == 3) and (rgb_img.shape[2] == 1):
      rgb_img = np.tile(rgb_img, [1, 1, 3])

    # Convert depth map to single channel if it is multichannel
    if (depth_map.ndim == 3) and depth_map.shape[2] == 3:
      depth_map = np.squeeze(depth_map[:, :, 0])
    depth_map = depth_map.astype(np.float32)

    # Get image dimensions
    H, W = depth_map.shape

    # Create meshgrid (pixel coordinates)
    Z = depth_map
    A, B = np.meshgrid(range(W), range(H))
    ones = np.ones_like(A)
    grid = np.concatenate((A[:, :, np.newaxis], B[:, :, np.newaxis], ones[:, :, np.newaxis]),
                          axis=2)
    grid = grid.astype(np.float32) * Z[:, :, np.newaxis]
    # Nx3 where each row is (a*Z, b*Z, Z)
    grid_flattened = grid.reshape((-1, 3))
    grid_flattened = grid_flattened.T  # 3xN where each col is (a*Z, b*Z, Z)
    prod = np.dot(self.K_inv, grid_flattened)
    XYZ = np.concatenate((prod[:2, :].T, Z.flatten()[:, np.newaxis]), axis=1)  # Nx3
    XYZRGB = np.hstack((XYZ, rgb_img.reshape((-1, 3))))
    points_3d = XYZRGB

    if labels is not None:
      labels_reshaped = labels.reshape((-1, n_label_channels))

    # Prune points
    if prune is True:
      valid = []
      for idx in range(points_3d.shape[0]):
        cur_y = points_3d[idx, 1]
        cur_z = points_3d[idx, 2]
        if (cur_z == 0) or (cur_z == 65535):  # Don't show things at 0 distance or max distance
          continue
        elif (max_depth is not None) and (cur_z > max_depth):
          continue
        elif (max_height is not None) and (cur_y > max_height):
          continue
        elif (min_height is not None) and (cur_y < min_height):
          continue
        else:
          valid.append(idx)
      points_3d = points_3d[np.asarray(valid)]
      if labels is not None:
        labels_reshaped = labels_reshaped[np.asarray(valid)]

    if extrinsics is not None:
      points_3d = self.camera2world(extrinsics, points_3d)

    if labels is not None:
      points_3d_labels = np.hstack((points_3d[:, :3], labels_reshaped))
      return points_3d, points_3d_labels
    else:
      return points_3d

  @staticmethod
  def _camera2world_transform(no_rgb_points_3d, R, T):
    points_3d_world = (np.dot(R.T, no_rgb_points_3d.T) - T).T  # Nx3
    return points_3d_world

  @staticmethod
  def _world2camera_transform(no_rgb_points_3d, R, T):
    points_3d_world = (np.dot(R, no_rgb_points_3d.T + T)).T  # Nx3
    return points_3d_world

  def _transform_points(self, points_3d, extrinsics, transform):
    """Base/wrapper method for transforming points using R and T.
    """
    assert points_3d.ndim == 2
    orig_points_3d = points_3d
    points_3d = np.copy(orig_points_3d)
    if points_3d.shape[1] == 6:  # XYZRGB
      points_3d = points_3d[:, :3]
    elif points_3d.shape[1] == 3:  # XYZ
      points_3d = points_3d
    else:
      raise ValueError('3D points need to be XYZ or XYZRGB.')

    R, T = self.extrinsics2RT(extrinsics)
    points_3d_world = transform(points_3d, R, T)

    # Add color again (if appropriate)
    if orig_points_3d.shape[1] == 6:  # XYZRGB
      points_3d_world = np.hstack((points_3d_world, orig_points_3d[:, -3:]))
    return points_3d_world

  def camera2world(self, extrinsics, points_3d):
    """Transform from camera coordinates (3D) to world coordinates (3D).

    Args:
      points_3d: Nx3 or Nx6 matrix of N points with XYZ or XYZRGB values.
    """
    return self._transform_points(points_3d, extrinsics, self._camera2world_transform)

  def world2camera(self, extrinsics, points_3d):
    """Transform from world coordinates (3D) to camera coordinates (3D).
    """
    return self._transform_points(points_3d, extrinsics, self._world2camera_transform)

  @property
  def intrinsics(self):
    return self._intrinsics

  @property
  def camera_matrix(self):
    return self._camera_matrix

  @property
  def K_inv(self):
    return self._K_inv


def sanitize_pointcloud(filepath_or_array):
  if isinstance(filepath_or_array, str):
    filepath = filepath_or_array
    if filepath.endswith('.ply'):
      pointcloud = read_plyfile(filepath)
    elif filepath.endswith('.txt'):
      pointcloud = np.loadtxt(filepath).astype(np.float32)
    else:
      raise ValueError('Unknown file format: ' + filepath)
  elif isinstance(filepath_or_array, np.ndarray):
    pointcloud = filepath_or_array.astype(np.float32)
  else:
    raise ValueError('Unknown pointcloud format')
  assert pointcloud.ndim == 2 and pointcloud.shape[1] >= 6
  return pointcloud[:, :6].astype(np.float32)


def colorize_pointcloud(xyz, label, ignore_label=-1, repeat_color=False):
  if not repeat_color:
    assert label[label != ignore_label].max() < len(COLOR_MAP_RGB), 'Not enough colors.'
  else:
    label %= len(COLOR_MAP_RGB)
  label_rgb = np.array([COLOR_MAP_RGB[i] if i != ignore_label else IGNORE_COLOR for i in label])
  return np.hstack((xyz, label_rgb))


class PlyWriter(object):

  POINTCLOUD_DTYPE = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                      ('blue', 'u1')]

  @classmethod
  def read_txt(cls, txtfile):
    # Read txt file and parse its content.
    with open(txtfile) as f:
      pointcloud = [l.split() for l in f]
    # Load point cloud to named numpy array.
    pointcloud = np.array(pointcloud).astype(np.float32)
    assert pointcloud.shape[1] == 6
    xyz = pointcloud[:, :3].astype(np.float32)
    rgb = pointcloud[:, 3:].astype(np.uint8)
    return xyz, rgb

  @staticmethod
  def write_ply(array, filepath):
    ply_el = PlyElement.describe(array, 'vertex')
    target_path, _ = os.path.split(filepath)
    if target_path != '' and not os.path.exists(target_path):
      os.makedirs(target_path)
    PlyData([ply_el]).write(filepath)

  @classmethod
  def write_vertex_only_ply(cls, vertices, filepath):
    # assume that points are N x 3 np array for vertex locations
    color = 255 * np.ones((len(vertices), 3))
    pc_points = np.array([tuple(p) for p in np.concatenate((vertices, color), axis=1)],
                         dtype=cls.POINTCLOUD_DTYPE)
    cls.write_ply(pc_points, filepath)

  @classmethod
  def write_ply_vert_color(cls, vertices, colors, filepath):
    # assume that points are N x 3 np array for vertex locations
    pc_points = np.array([tuple(p) for p in np.concatenate((vertices, colors), axis=1)],
                         dtype=cls.POINTCLOUD_DTYPE)
    cls.write_ply(pc_points, filepath)

  @classmethod
  def concat_label(cls, target, xyz, label):
    subpointcloud = np.concatenate([xyz, label], axis=1)
    subpointcloud = np.array([tuple(l) for l in subpointcloud], dtype=cls.POINTCLOUD_DTYPE)
    return np.concatenate([target, subpointcloud], axis=0)


def revert_batch_to_ply(batch):
  """Transforms a batch back into a ply format.
  Used for visual debugging of the data loader."""
  import seaborn as sns
  import numpy as np

  LABEL_COLORMAP = sns.color_palette('hls', 16)
  data, target = batch
  clean_end = False
  for batch_idx, datum in enumerate(data):
    clean_end = False
    xyz, rgb = datum
    label = target[:len(xyz)]
    xyz = xyz.cpu().numpy()
    rgb = rgb.cpu().numpy()
    label = label.cpu().numpy().astype(int)
    template = np.array([], dtype=PlyWriter.POINTCLOUD_DTYPE)
    xyzrgb = PlyWriter.concat_label(template, xyz, rgb)
    label2color = np.array([LABEL_COLORMAP[i] for i in label])
    xyzlabel = PlyWriter.concat_label(template, xyz, label2color)
    PlyWriter.write_ply(xyzrgb, osp.join(os.getcwd(), 'out_%02d_rgb.ply' % batch_idx))
    PlyWriter.write_ply(xyzlabel, osp.join(os.getcwd(), 'out_%02d_label.ply' % batch_idx))
    if len(target) == len(xyz):
      clean_end = True
    else:
      target = target[len(xyz):]
  assert clean_end, '# label =/= # rgb points'


def get_bbox(coords, semantic_labels, instance_labels, instance_mask, invalid_label,
             strict_semantic_match=True, is_voxel=True):
  """Generate surrounding bounding boxes from instance labels."""
  instances_fg = instance_labels[instance_mask]
  semantic_fg = semantic_labels[instance_mask]
  coords_fg = coords[instance_mask]
  instance_labels_new = np.ones_like(instance_labels) * invalid_label
  bboxes = []
  instance_ids, instance_mapping = np.unique(instances_fg, return_inverse=True)
  instance_labels_new[instance_mask] = instance_mapping
  for instance_id in np.unique(instance_ids):
    instance_map = instances_fg == instance_id
    coords_inst = coords_fg[instance_map]
    semantic_sub_labels, semantic_counts = np.unique(semantic_fg[instance_map], return_counts=True)
    if strict_semantic_match and semantic_sub_labels.size != 1:
      raise ValueError('Mismatch between semantic and instance label.')
    semantic_class = semantic_sub_labels[np.argmax(semantic_counts)]
    bbox_min = np.min(coords_inst, 0)
    if is_voxel:
      bbox_min -= 0.5
    bbox_max = np.max(coords_inst, 0)
    if is_voxel:
      bbox_max += 0.5
    bboxes.append(np.concatenate((bbox_min, bbox_max, np.array([semantic_class]))))
  if not bboxes:
    return np.empty((0, 7)), instance_labels_new
  return np.vstack(bboxes), instance_labels_new


def bboxes2corners(bboxes, bbox_param='xyzxyz', rot_axis='z', swap_yz=False):
    def _bbox2corners(bbox_size, bbox_center, bbox_rot, rot_axis='z', swap_yz=False):
      rot_c = np.cos(bbox_rot)
      rot_s = np.sin(bbox_rot)
      if rot_axis == 'x':
        rot = np.array([[1, 0, 0],
                        [0, rot_c, -rot_s],
                        [0, rot_s, rot_c]])
      elif rot_axis == 'y':
        rot = np.array([[rot_c, 0, rot_s],
                        [0, 1, 0],
                        [-rot_s, 0, rot_c]])
      elif rot_axis == 'z':
        rot = np.array([[rot_c, -rot_s, 0],
                        [rot_s, rot_c, 0],
                        [0, 0, 1]])
      else:
        raise ValueError(f'Unknown rotation axis {rot_axis}')
      xs, ys, zs = bbox_size
      xc = np.stack([
          xs / 2, xs / 2, -xs / 2, -xs / 2, xs / 2, xs / 2, -xs / 2, -xs / 2
      ])
      if swap_yz:
        yx = np.stack([
            ys / 2, -ys / 2, -ys / 2, ys / 2, ys / 2, -ys / 2, -ys / 2, ys / 2
        ])
        zc = np.stack([
            zs / 2, zs / 2, zs / 2, zs / 2, -zs / 2, -zs / 2, -zs / 2, -zs / 2
        ])
      else:
        yx = np.stack([
            ys / 2, ys / 2, ys / 2, ys / 2, -ys / 2, -ys / 2, -ys / 2, -ys / 2
        ])
        zc = np.stack([
            zs / 2, -zs / 2, -zs / 2, zs / 2, zs / 2, -zs / 2, -zs / 2, zs / 2
        ])
      bbox_xyz = (rot @ np.stack((xc, yx, zc))).T + bbox_center
      if swap_yz:
        bbox_xyz[:, (1, 2)] = bbox_xyz[:, (2, 1)]
      return bbox_xyz
    if bbox_param == 'corners':
      return bboxes
    if len(bboxes.shape) == 1:
      bboxes = np.array([bboxes])
    if len(bboxes.shape) != 2:
      raise ValueError('Unknown bounding box shape')
    num_bbox = bboxes.shape[0]
    bbox_rot = np.zeros(num_bbox)
    if bbox_param in ('xyzxyzr', 'xyzwhlr'):
      if bboxes.shape[1] != 7:
        raise ValueError('Unknown bounding box shape')
      bbox_rot = bboxes[:, 6]
    elif bboxes.shape[1] != 6:
      raise ValueError('Unknown bounding box shape')
    if bbox_param in ('xyzxyz', 'xyzxyzr'):
      bbox_center = (bboxes[:, 3:6] + bboxes[:, :3]) / 2
      bbox_size = bboxes[:, 3:6] - bboxes[:, :3]
    elif bbox_param in ('xyzwhl', 'xyzwhlr'):
      bbox_center = bboxes[:, :3]
      bbox_size = bboxes[:, 3:6]
    else:
      raise ValueError(f'Unknown bounding box parameterzation: {bbox_param}')
    if not np.all(bbox_size > 0):
      raise ValueError('Not all bouding box size positive')
    return np.stack([_bbox2corners(*params, rot_axis=rot_axis, swap_yz=swap_yz)
                     for params in zip(bbox_size, bbox_center, bbox_rot)])


def visualize_pcd(*pcds, camera_params=None, save_image=None):
  def _convert2o3d(xyz):
    if not isinstance(xyz, np.ndarray):
      return xyz
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
    if xyz.shape[1] == 6 and xyz.shape[0] > 0:
      colors = xyz[:, 3:]
      if colors.max() > 1:
          colors /= 255
      colors = np.clip(colors, 0., 1.)
      pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
  pcds = [_convert2o3d(x) for x in pcds]
  vis = o3d.visualization.Visualizer()
  vis.create_window()
  for pcd in pcds:
    vis.add_geometry(pcd)
  ctr = vis.get_view_control()
  if camera_params is not None:
    ctr.convert_from_pinhole_camera_parameters(camera_params)
  vis.run()
  if save_image:
    image = vis.capture_screen_float_buffer(False)
    plt.imsave(save_image, np.asarray(image), dpi=1)
  param = vis.get_view_control().convert_to_pinhole_camera_parameters()
  return param


def visualize_bboxes(bboxes, bboxes_cls=None, bbox_param='xyzxyz', rot_axis='z', num_points=30):
  """Generate [number of bounding boxes] x 6 colored point cloud matrix of 3D bounding boxes.

  Args:
    bboxes: [number of bounding boxes] x 6 matrix center coordinate of the bounding boxes.
    bboxes_cls: [number of bounding boxes] semantic label of bounding boxes.
  """
  def _vector_linspace(vector_a, vector_b, num_points=30):
    vector_diff = (np.array(vector_a) - vector_b) / (num_points - 1)
    out = np.tile(vector_a, (num_points, 1))
    for i in range(1, num_points):
      out[i:] -= vector_diff
    return out

  if bboxes_cls is None:
    bboxes_cls = np.arange(bboxes.shape[0])
  if bboxes.shape[0] == 0:
    return np.zeros((0, 6))

  bbox_corners = bboxes2corners(bboxes, bbox_param=bbox_param, rot_axis=rot_axis)

  label_vector = []
  bboxes_pts = np.empty((0, 3))
  for bbox, bbox_cls in zip(bbox_corners, bboxes_cls):
    bbox_subptc = []
    for i in range(4):
      bbox_subptc.extend([
          _vector_linspace(bbox[i], bbox[(i + 1) % 4], num_points=num_points),
          _vector_linspace(bbox[i + 4], bbox[(i + 1) % 4 + 4], num_points=num_points),
          _vector_linspace(bbox[i], bbox[i + 4], num_points=num_points),
      ])
    bbox_pts = np.vstack(bbox_subptc)
    bboxes_pts = np.vstack((bboxes_pts, bbox_pts))
    label_vector.extend([int(bbox_cls)] * bbox_pts.shape[0])
  return colorize_pointcloud(bboxes_pts, np.array(label_vector), repeat_color=True)


def test_point_cloud_generation():
  import numpy as np
  import pickle
  from lib.datasets.synthia import SynthiaDataset

  with open('/tmp/yo.p', 'r') as f:
    data_list = pickle.load(f)
  data_list = sorted(data_list, key=lambda k: k['Stereo_Right']['Omni_F']['rgb'])

  all_points_3d = []
  start_frame = 100
  num_frames = 1
  for idx, frame_count in enumerate(range(num_frames)):
    print('Index:', idx)
    freq = 15
    frame_idx = start_frame + frame_count * freq  # Backproject every freq frames
    data_dict = data_list[frame_idx]
    for stereo_dir in data_dict.keys():
      for omni_dir in data_dict[stereo_dir].keys():
        data = data_dict[stereo_dir][omni_dir]
        intrinsics = SynthiaDataset.load_intrinsics('/cvgl/group/Synthia/' + data['intrinsics'])
        extrinsics = SynthiaDataset.load_extrinsics('/cvgl/group/Synthia/' + data['extrinsics'])
        rgb = SynthiaDataset.load_rgb('/cvgl/group/Synthia/' + data['rgb'])
        depth = SynthiaDataset.load_depth('/cvgl/group/Synthia/' + data['depth'])
        camera = Camera(intrinsics)
        points_3d = camera.backproject(
            depth, max_depth=2000., max_height=500., rgb_img=rgb, extrinsics=extrinsics)

        # Subsample
        points_3d = points_3d[::5]
        all_points_3d.append(points_3d)
  all_points_3d = np.vstack(all_points_3d)
  save_point_cloud(all_points_3d, '/tmp/test_point_cloud.ply')


if __name__ == '__main__':
  test_point_cloud_generation()
