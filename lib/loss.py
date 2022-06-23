import numpy as np
import torch
import torch.nn as nn


class RotationLoss(nn.Module):
  def __init__(self, num_rotation_bins, activation='none', min_angle=-np.pi, max_angle=np.pi):
    super().__init__()
    if activation == 'none':
      self.activation_fn = None
    elif activation == 'tanh':
      self.activation_fn = torch.tanh
    elif activation == 'sigmoid':
      self.activation_fn = torch.sigmoid
    self.num_rotation_bins = num_rotation_bins
    self.min_angle = min_angle
    self.max_angle = max_angle

  def _activate(self, output):
    if self.activation_fn is not None:
      return self.activation_fn(output)
    return output


class RotationCircularLoss(RotationLoss):

  NUM_OUTPUT = 2

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.loss = nn.MSELoss()

  def pred(self, output):
    output = self._activate(output)
    return torch.atan2(output[..., 0], output[..., 1])

  def forward(self, output, target):
    output = self._activate(output)
    return self.loss(output, torch.stack((torch.sin(target), torch.cos(target))).T)


class RotationClassificationLoss(RotationLoss):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.class_criterion = nn.CrossEntropyLoss()
    self.NUM_OUTPUT = self.num_rotation_bins
    self.angle_per_class = (self.max_angle - self.min_angle) / float(self.num_rotation_bins)

  def pred(self, output):
    if output.shape[0]:
      return output.argmax(1) * self.angle_per_class + self.min_angle + self.angle_per_class / 2
    return torch.zeros(0).to(output)

  def forward(self, output, target):
    target2class = torch.clamp(
        (target - self.min_angle) // self.angle_per_class, 0, self.num_rotation_bins - 1)
    return self.class_criterion(output, target2class.long())


class RotationErrorLoss1(RotationLoss):

  NUM_OUTPUT = 2

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.loss = nn.L1Loss()

  def pred(self, output):
    output = self._activate(output)
    return torch.atan2(output[..., 0], output[..., 1])

  def forward(self, output, target):
    return self.loss(self.pred(output), target)


class RotationErrorLoss2(RotationLoss):

  NUM_OUTPUT = 2

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def pred(self, output):
    output = self._activate(output)
    return torch.atan2(output[..., 0], output[..., 1])

  def forward(self, output, target):
    output = self._activate(output)
    target = torch.stack((torch.sin(target), torch.cos(target))).T
    side = (target * output).sum(1)
    return torch.acos(torch.clamp(side, min=-0.999, max=0.999)).mean()


ROT_LOSS_NAME2CLASS = {
    'circular': RotationCircularLoss,
    'classification': RotationClassificationLoss,
    'rotationerror1': RotationErrorLoss1,
    'rotationerror2': RotationErrorLoss2,
}


def get_rotation_loss(loss_name):
  return ROT_LOSS_NAME2CLASS[loss_name]


class FocalLoss(nn.Module):
  def __init__(self, alpha=0.25, gamma=2, reduction='mean', ignore_index=-1):
    super(FocalLoss, self).__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction
    self.ignore_lb = ignore_index
    self.crit = nn.BCEWithLogitsLoss(reduction='none')

  def forward(self, logits, label):
    '''
    args: logits: tensor of shape (N, C, H, W)
    args: label: tensor of shape(N, H, W)
    '''
    # overcome ignored label
    with torch.no_grad():
      label = label.clone().detach()
      ignore = label == self.ignore_lb
      n_valid = (ignore == 0).sum()
      label[ignore] = 0
      lb_one_hot = torch.zeros_like(logits).scatter_(
          1, label.unsqueeze(1), 1).detach()
      alpha = torch.empty_like(logits).fill_(1 - self.alpha)
      alpha[lb_one_hot == 1] = self.alpha

    # compute loss
    probs = torch.sigmoid(logits)
    pt = torch.where(lb_one_hot == 1, probs, 1 - probs)
    ce_loss = self.crit(logits, lb_one_hot)
    loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss).sum(dim=1)
    loss[ignore == 1] = 0
    if self.reduction == 'mean':
      loss = loss.sum() / n_valid
    if self.reduction == 'sum':
      loss = loss.sum()
    return loss


class BalancedLoss(nn.Module):
  NUM_LABELS = 2

  def __init__(self, ignore_index=-1):
    super().__init__()
    self.ignore_index = ignore_index
    self.crit = nn.CrossEntropyLoss(ignore_index=ignore_index)

  def forward(self, logits, label):
    assert torch.all(label < self.NUM_LABELS)
    loss = torch.scalar_tensor(0.).to(logits)
    for i in range(self.NUM_LABELS):
      target_mask = label == i
      if torch.any(target_mask):
        loss += self.crit(logits[target_mask], label[target_mask]) / self.NUM_LABELS
    return loss


CLASSIFICATION_LOSS_NAME2CLASS = {
    'focal': FocalLoss,
    'balanced': BalancedLoss,
    'ce': nn.CrossEntropyLoss,
}


def get_classification_loss(loss_name):
  return CLASSIFICATION_LOSS_NAME2CLASS[loss_name]
