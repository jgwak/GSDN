## SYNTHIA Point Cloud Dataset

### Step 1: Build the paths

In this step, we build the relative paths of the extrinsics, intrinsics, depth, rgb, etc. files and
store them as a list of dicts, one dict per frame. We can build and write the pickle file using
the `build_data_paths` function in `synthia.py`. By default, this is saved as `synthia.p`.
```
python -m lib.datasets.synthia
```

The original reason for storing this file was for on-the-fly point cloud/image generation. Now we
use it as input for our point cloud/octree writer.

```
with open('/cvgl/group/Synthia/synthia-processed/synthia.p', 'rb') as f:
  data_list = pickle.load(f)

pprint(data_list[0])
{'Stereo_Left': {'Omni_B': {'depth': 'SYNTHIA-SEQS-01-DAWN/Depth/Stereo_Left/Omni_B/000000.png',
                            'extrinsics': 'SYNTHIA-SEQS-01-DAWN/CameraParams/Stereo_Left/Omni_B/000000.txt',
                            'intrinsics': 'SYNTHIA-SEQS-01-DAWN/CameraParams/intrinsics.txt',
                            'label_color': 'SYNTHIA-SEQS-01-DAWN/GT/COLOR/Stereo_Left/Omni_B/000000.png',
                            'label_idx': 'SYNTHIA-SEQS-01-DAWN/GT/LABELS/Stereo_Left/Omni_B/000000.png',
                            'rgb': 'SYNTHIA-SEQS-01-DAWN/RGB/Stereo_Left/Omni_B/000000.png'},
                 'Omni_F': {'depth': 'SYNTHIA-SEQS-01-DAWN/Depth/Stereo_Left/Omni_F/000000.png',
                            'extrinsics': 'SYNTHIA-SEQS-01-DAWN/CameraParams/Stereo_Left/Omni_F/000000.txt',
                            'intrinsics': 'SYNTHIA-SEQS-01-DAWN/CameraParams/intrinsics.txt',
                            'label_color': 'SYNTHIA-SEQS-01-DAWN/GT/COLOR/Stereo_Left/Omni_F/000000.png',
                            'label_idx': 'SYNTHIA-SEQS-01-DAWN/GT/LABELS/Stereo_Left/Omni_F/000000.png',
                            'rgb': 'SYNTHIA-SEQS-01-DAWN/RGB/Stereo_Left/Omni_F/000000.png'},
                 'Omni_L': {'depth': 'SYNTHIA-SEQS-01-DAWN/Depth/Stereo_Left/Omni_L/000000.png',
                            'extrinsics': 'SYNTHIA-SEQS-01-DAWN/CameraParams/Stereo_Left/Omni_L/000000.txt',
                            'intrinsics': 'SYNTHIA-SEQS-01-DAWN/CameraParams/intrinsics.txt',
                            'label_color': 'SYNTHIA-SEQS-01-DAWN/GT/COLOR/Stereo_Left/Omni_L/000000.png',
                            'label_idx': 'SYNTHIA-SEQS-01-DAWN/GT/LABELS/Stereo_Left/Omni_L/000000.png',
                            'rgb': 'SYNTHIA-SEQS-01-DAWN/RGB/Stereo_Left/Omni_L/000000.png'},
                 'Omni_R': {'depth': 'SYNTHIA-SEQS-01-DAWN/Depth/Stereo_Left/Omni_R/000000.png',
                            'extrinsics': 'SYNTHIA-SEQS-01-DAWN/CameraParams/Stereo_Left/Omni_R/000000.txt',
                            'intrinsics': 'SYNTHIA-SEQS-01-DAWN/CameraParams/intrinsics.txt',
                            'label_color': 'SYNTHIA-SEQS-01-DAWN/GT/COLOR/Stereo_Left/Omni_R/000000.png',
                            'label_idx': 'SYNTHIA-SEQS-01-DAWN/GT/LABELS/Stereo_Left/Omni_R/000000.png',
                            'rgb': 'SYNTHIA-SEQS-01-DAWN/RGB/Stereo_Left/Omni_R/000000.png'}},
 'Stereo_Right': {'Omni_B': {'depth': 'SYNTHIA-SEQS-01-DAWN/Depth/Stereo_Right/Omni_B/000000.png',
                             'extrinsics': 'SYNTHIA-SEQS-01-DAWN/CameraParams/Stereo_Right/Omni_B/000000.txt',
                             'intrinsics': 'SYNTHIA-SEQS-01-DAWN/CameraParams/intrinsics.txt',
                             'label_color': 'SYNTHIA-SEQS-01-DAWN/GT/COLOR/Stereo_Right/Omni_B/000000.png',
                             'label_idx': 'SYNTHIA-SEQS-01-DAWN/GT/LABELS/Stereo_Right/Omni_B/000000.png',
                             'rgb': 'SYNTHIA-SEQS-01-DAWN/RGB/Stereo_Right/Omni_B/000000.png'},
                  'Omni_F': {'depth': 'SYNTHIA-SEQS-01-DAWN/Depth/Stereo_Right/Omni_F/000000.png',
                             'extrinsics': 'SYNTHIA-SEQS-01-DAWN/CameraParams/Stereo_Right/Omni_F/000000.txt',
                             'intrinsics': 'SYNTHIA-SEQS-01-DAWN/CameraParams/intrinsics.txt',
                             'label_color': 'SYNTHIA-SEQS-01-DAWN/GT/COLOR/Stereo_Right/Omni_F/000000.png',
                             'label_idx': 'SYNTHIA-SEQS-01-DAWN/GT/LABELS/Stereo_Right/Omni_F/000000.png',
                             'rgb': 'SYNTHIA-SEQS-01-DAWN/RGB/Stereo_Right/Omni_F/000000.png'},
                  'Omni_L': {'depth': 'SYNTHIA-SEQS-01-DAWN/Depth/Stereo_Right/Omni_L/000000.png',
                             'extrinsics': 'SYNTHIA-SEQS-01-DAWN/CameraParams/Stereo_Right/Omni_L/000000.txt',
                             'intrinsics': 'SYNTHIA-SEQS-01-DAWN/CameraParams/intrinsics.txt',
                             'label_color': 'SYNTHIA-SEQS-01-DAWN/GT/COLOR/Stereo_Right/Omni_L/000000.png',
                             'label_idx': 'SYNTHIA-SEQS-01-DAWN/GT/LABELS/Stereo_Right/Omni_L/000000.png',
                             'rgb': 'SYNTHIA-SEQS-01-DAWN/RGB/Stereo_Right/Omni_L/000000.png'},
                  'Omni_R': {'depth': 'SYNTHIA-SEQS-01-DAWN/Depth/Stereo_Right/Omni_R/000000.png',
                             'extrinsics': 'SYNTHIA-SEQS-01-DAWN/CameraParams/Stereo_Right/Omni_R/000000.txt',
                             'intrinsics': 'SYNTHIA-SEQS-01-DAWN/CameraParams/intrinsics.txt',
                             'label_color': 'SYNTHIA-SEQS-01-DAWN/GT/COLOR/Stereo_Right/Omni_R/000000.png',
                             'label_idx': 'SYNTHIA-SEQS-01-DAWN/GT/LABELS/Stereo_Right/Omni_R/000000.png',
                             'rgb': 'SYNTHIA-SEQS-01-DAWN/RGB/Stereo_Right/Omni_R/000000.png'}}}
```

### Step 2: Write/cache the point clouds

We can use the `SynthiaOctreeBuilder` class to write the point clouds to a destination root
directory. When finished, the `SynthiaOctreeBuilder` will also write a pickle file `pc_paths.p` in
the destination root directory. This pickle file contains the (relative) paths to each point cloud.
Note that each point cloud is stored as a binary PLY file by default.

To write the point clouds:
```
python -m lib.datasets.synthia_octree_builder
```

Note that this saves the raw (unclipped and not downsampled) point clouds. Make sure
config.synthia_path is set to the destination root directory for the point cloud files if
you want to use the data loader to read the `SynthiaPCDataset` as raw point clouds.

### Step 3: Downsample and clip

Run `voxel_grid [directory]` to voxelize files in a given directory. See the
`voxelization/README.md` for more details.
