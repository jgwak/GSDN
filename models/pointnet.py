import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import MinkowskiEngine as ME


class STN3d(nn.Module):
  def __init__(self, D=3):
    super(STN3d, self).__init__()
    self.conv1 = nn.Conv1d(3, 64, 1, bias=False)
    self.conv2 = nn.Conv1d(64, 128, 1, bias=False)
    self.conv3 = nn.Conv1d(128, 256, 1, bias=False)
    self.fc1 = nn.Linear(256, 128, bias=False)
    self.fc2 = nn.Linear(128, 64, bias=False)
    self.fc3 = nn.Linear(64, 9)
    self.relu = nn.ReLU()
    self.pool = ME.MinkowskiGlobalPooling()

    self.bn1 = nn.BatchNorm1d(64)
    self.bn2 = nn.BatchNorm1d(128)
    self.bn3 = nn.BatchNorm1d(256)
    self.bn4 = nn.BatchNorm1d(128)
    self.bn5 = nn.BatchNorm1d(64)
    self.broadcast = ME.MinkowskiBroadcast()

  def forward(self, x):
    xf = self.relu(self.bn1(self.conv1(x.F.unsqueeze(-1))[..., 0]).unsqueeze(-1))
    xf = self.relu(self.bn2(self.conv2(xf)[..., 0]).unsqueeze(-1))
    xf = self.relu(self.bn3(self.conv3(xf)[..., 0]).unsqueeze(-1))
    xf = ME.SparseTensor(xf[..., 0], coords_key=x.coords_key, coords_manager=x.coords_man)
    xfc = self.pool(xf)

    xf = F.relu(self.bn4(self.fc1(self.pool(xfc).F)))
    xf = F.relu(self.bn5(self.fc2(xf)))
    xf = self.fc3(xf)
    xf += torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0, 1]],
                       dtype=x.dtype, device=x.device).repeat(xf.shape[0], 1)
    xf = ME.SparseTensor(xf, coords_key=xfc.coords_key, coords_manager=xfc.coords_man)
    xfc = ME.SparseTensor(torch.zeros(x.shape[0], 9, dtype=x.dtype, device=x.device),
                          coords_key=x.coords_key, coords_manager=x.coords_man)
    return self.broadcast(xfc, xf)


class PointNetfeat(nn.Module):

  def __init__(self, in_channels):
    super(PointNetfeat, self).__init__()
    self.pool = ME.MinkowskiGlobalPooling()
    self.broadcast = ME.MinkowskiBroadcast()
    self.stn = STN3d(D=3)
    self.conv1 = nn.Conv1d(in_channels + 3, 128, 1, bias=False)
    self.conv2 = nn.Conv1d(128, 256, 1, bias=False)
    self.bn1 = nn.BatchNorm1d(128)
    self.bn2 = nn.BatchNorm1d(256)
    self.relu = nn.ReLU()

  def forward(self, x):
    # First, align coordinates to be centered around zero.
    coords = x.coords.to(x.device)[:, 1:]
    coords = ME.SparseTensor(coords.float(), coords_key=x.coords_key, coords_manager=x.coords_man)
    mean_coords = self.broadcast(coords, self.pool(coords))
    norm_coords = coords - mean_coords
    # Second, apply spatial transformer to the coordinates.
    trans = self.stn(norm_coords)
    coords_feat_stn = torch.squeeze(torch.bmm(norm_coords.F.view(-1, 1, 3), trans.F.view(-1, 3, 3)))
    xf = torch.cat((coords_feat_stn, x.F), 1).unsqueeze(-1)
    xf = self.relu(self.bn1(self.conv1(xf)[..., 0]).unsqueeze(-1))

    pointfeat = xf
    xf = self.bn2(self.conv2(xf)[..., 0]).unsqueeze(-1)
    xfc = ME.SparseTensor(xf[..., 0], coords_key=x.coords_key, coords_manager=x.coords_man)
    xf_avg = ME.SparseTensor(
        torch.zeros(x.shape[0], xfc.F.shape[1], dtype=x.dtype, device=x.device),
        coords_key=x.coords_key, coords_manager=x.coords_man)
    xf_avg = self.broadcast(xf_avg, self.pool(xfc))
    return torch.cat((pointfeat[..., 0], xf_avg.F), 1)


class PointNet(nn.Module):
  OUT_PIXEL_DIST = 1

  def __init__(self, in_channels, out_channels, config, D=3, return_feat=False, **kwargs):
    super(PointNet, self).__init__()
    self.k = out_channels
    self.feat = PointNetfeat(in_channels)
    self.conv1 = nn.Conv1d(384, 128, 1, bias=False)
    self.conv2 = nn.Conv1d(128, 64, 1, bias=False)
    self.conv3 = nn.Conv1d(64, self.k, 1)
    self.bn1 = nn.BatchNorm1d(128)
    self.bn2 = nn.BatchNorm1d(64)
    self.relu = nn.ReLU()

  def forward(self, x):
    coords_key, coords_manager = x.coords_key, x.coords_man
    x = self.feat(x)
    x = self.relu(self.bn1(self.conv1(x.unsqueeze(-1))[..., 0]).unsqueeze(-1))
    x = self.relu(self.bn2(self.conv2(x)[..., 0]).unsqueeze(-1))
    x = self.conv3(x)
    return ME.SparseTensor(x.squeeze(-1), coords_key=coords_key, coords_manager=coords_manager)


class PointNetXS(nn.Module):
  OUT_PIXEL_DIST = 1

  def __init__(self, in_channels, out_channels, config, D=3, return_feat=False, **kwargs):
    super().__init__()
    self.k = out_channels
    self.conv1 = nn.Conv1d(in_channels, 128, 1, bias=False)
    self.conv2 = nn.Conv1d(256, 64, 1, bias=False)
    self.conv3 = nn.Conv1d(64, self.k, 1)

    self.bn1 = nn.BatchNorm1d(128)
    self.bn2 = nn.BatchNorm1d(64)
    self.relu = nn.ReLU()

  def forward(self, x):
    batch_idx, coords_key, coords_manager = x.C[:, 0], x.coords_key, x.coords_man
    unique_batch_idx = torch.unique(batch_idx)
    x = self.bn1(self.conv1(x.F.unsqueeze(-1)))
    max_x = [x[batch_idx == i].max(0)[0].unsqueeze(0).expand((batch_idx == i).sum(), -1, -1)
             for i in unique_batch_idx]
    x = torch.cat((x, torch.cat(max_x, 0)), 1)
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.conv3(x)
    return ME.SparseTensor(x.squeeze(-1), coords_key=coords_key, coords_manager=coords_manager)
