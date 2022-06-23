import glob
import random
import os
import numpy as np

import tqdm

DET_BASE_DIR = '/cvgl2/u/jgwak/Datasets/sunrgbd/detection'
TRAIN_RATIO = 0.8

trainval_list = []
test_list = []

for fn in tqdm.tqdm(list(glob.glob(os.path.join(DET_BASE_DIR, 'train/*_pc.npz')))):
  pc = dict(np.load(fn))['pc']
  fns = fn.split(os.sep)
  fid = fns[-1].split('_')[0]
  bbox = np.load('/'.join(fns[:-1] + [fid + '_bbox.npy']))
  bbox = np.hstack((bbox[:, :3] - bbox[:, 3:6], bbox[:, :3] + bbox[:, 3:6], bbox[:, 6:]))
  new_fid = f'train/{fid}.npz'
  np.savez_compressed(new_fid, pc=pc, bbox=bbox)
  trainval_list.append(new_fid)

for fn in tqdm.tqdm(list(glob.glob(os.path.join(DET_BASE_DIR, 'val/*_pc.npz')))):
  pc = dict(np.load(fn))['pc']
  fns = fn.split(os.sep)
  fid = fns[-1].split('_')[0]
  bbox = np.load('/'.join(fns[:-1] + [fid + '_bbox.npy']))
  bbox = np.hstack((bbox[:, :3] - bbox[:, 3:6], bbox[:, :3] + bbox[:, 3:6], bbox[:, 6:]))
  new_fid = f'test/{fid}.npz'
  np.savez_compressed(new_fid, pc=pc, bbox=bbox)
  test_list.append(new_fid)

random.seed(1)
random.shuffle(trainval_list)
numtrain = int(len(trainval_list) * TRAIN_RATIO)
train_list = trainval_list[:numtrain]
val_list = trainval_list[numtrain:]


def write_list(fn, fl):
  with open(fn, 'w') as f:
    f.writelines('\n'.join(fl))


write_list('train.txt', train_list)
write_list('val.txt', val_list)
write_list('trainval.txt', trainval_list)
write_list('test.txt', test_list)
