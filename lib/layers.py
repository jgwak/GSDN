import torch
import torch.nn as nn

from MinkowskiEngine import MinkowskiGlobalPooling, MinkowskiBroadcastAddition, MinkowskiBroadcastMultiplication


class MinkowskiInstanceNorm(nn.Module):

  def __init__(self, num_features, eps=1e-5, D=-1):
    super(MinkowskiInstanceNorm, self).__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(1, num_features))
    self.bias = nn.Parameter(torch.zeros(1, num_features))

    self.mean_in = MinkowskiGlobalPooling(dimension=D)
    self.glob_sum = MinkowskiBroadcastAddition(dimension=D)
    self.glob_sum2 = MinkowskiBroadcastAddition(dimension=D)
    self.glob_mean = MinkowskiGlobalPooling(dimension=D)
    self.glob_times = MinkowskiBroadcastMultiplication(dimension=D)
    self.D = D
    self.reset_parameters()

  def __repr__(self):
    s = f'(pixel_dist={self.pixel_dist}, D={self.D})'
    return self.__class__.__name__ + s

  def reset_parameters(self):
    self.weight.data.fill_(1)
    self.bias.data.zero_()

  def _check_input_dim(self, input):
    if input.dim() != 2:
      raise ValueError('expected 2D input (got {}D input)'.format(input.dim()))

  def forward(self, x):
    self._check_input_dim(x)
    mean_in = self.mean_in(x)
    temp = self.glob_sum(x, -mean_in)**2
    var_in = self.glob_mean(temp.data)
    instd_in = 1 / (var_in + self.eps).sqrt()

    x = self.glob_times(self.glob_sum2(x, -mean_in), instd_in)
    return x * self.weight + self.bias


class SwitchNorm1d(nn.Module):
  """
  https://github.com/switchablenorms/Switchable-Normalization/blob/master/models/switchable_norm.py
  """

  def __init__(self, num_features, eps=1e-5, momentum=0.997):
    super(SwitchNorm1d, self).__init__()
    self.eps = eps
    self.momentum = momentum
    self.weight = nn.Parameter(torch.ones(1, num_features))
    self.bias = nn.Parameter(torch.zeros(1, num_features))
    self.mean_weight = nn.Parameter(torch.ones(2))
    self.var_weight = nn.Parameter(torch.ones(2))
    self.register_buffer('running_mean', torch.zeros(1, num_features))
    self.register_buffer('running_var', torch.zeros(1, num_features))
    self.reset_parameters()

  def reset_parameters(self):
    self.running_mean.zero_()
    self.running_var.zero_()
    self.weight.data.fill_(1)
    self.bias.data.zero_()

  def _check_input_dim(self, input):
    if input.dim() != 2:
      raise ValueError('expected 2D input (got {}D input)'.format(input.dim()))

  def forward(self, x):
    self._check_input_dim(x)
    mean_ln = x.mean(1, keepdim=True)
    var_ln = x.var(1, keepdim=True)

    if self.training:
      mean_bn = x.mean(0, keepdim=True)
      var_bn = x.var(0, keepdim=True)
      self.running_mean.mul_(self.momentum)
      self.running_mean.add_((1 - self.momentum) * mean_bn.data)
      self.running_var.mul_(self.momentum)
      self.running_var.add_((1 - self.momentum) * var_bn.data)
    else:
      mean_bn = torch.autograd.Variable(self.running_mean)
      var_bn = torch.autograd.Variable(self.running_var)

    softmax = nn.Softmax(dim=0)
    mean_weight = softmax(self.mean_weight)
    var_weight = softmax(self.var_weight)

    mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
    var = var_weight[0] * var_ln + var_weight[1] * var_bn

    x = (x - mean) / (var + self.eps).sqrt()
    return x * self.weight + self.bias


class MinkowskiSwitchNorm(nn.Module):

  def __init__(self, num_features, eps=1e-5, momentum=0.99, last_gamma=False, D=-1):
    super(MinkowskiSwitchNorm, self).__init__()
    self.eps = eps
    self.momentum = momentum
    self.last_gamma = last_gamma
    self.weight = nn.Parameter(torch.ones(1, num_features))
    self.bias = nn.Parameter(torch.zeros(1, num_features))
    self.mean_weight = nn.Parameter(torch.ones(3))
    self.var_weight = nn.Parameter(torch.ones(3))
    self.register_buffer('running_mean', torch.zeros(1, num_features))
    self.register_buffer('running_var', torch.zeros(1, num_features))

    self.mean_in = MinkowskiGlobalPooling(dimension=D)
    self.glob_sum = MinkowskiBroadcastAddition(dimension=D)
    self.glob_sum2 = MinkowskiBroadcastAddition(dimension=D)
    self.glob_mean = MinkowskiGlobalPooling(dimension=D)
    self.glob_times = MinkowskiBroadcastMultiplication(dimension=D)
    self.D = D

    self.reset_parameters()

  def __repr__(self):
    s = f'(D={self.D})'
    return self.__class__.__name__ + s

  def sparse_global_mean(self, x):
    return MinkowskiGlobalPooling(dimension=self.D)(x)

  def sparse_global_sum(self, x, y):
    return MinkowskiBroadcastAddition(dimension=self.D)(x, y)

  def sparse_global_times(self, x, y):
    return MinkowskiBroadcastMultiplication(dimension=self.D)(x, y)

  def reset_parameters(self):
    self.reset_running_stat()
    if self.last_gamma:
      self.weight.data.fill_(0)
    else:
      self.weight.data.fill_(1)
    self.bias.data.zero_()

  def reset_running_stat(self):
    self.running_mean.zero_()
    self.running_var.zero_()

  def _check_input_dim(self, input):
    if input.dim() != 2:
      raise ValueError('expected 2D input (got {}D input)'.format(input.dim()))

  def forward(self, x):
    self._check_input_dim(x)
    mean_in = self.mean_in(x)
    temp = self.glob_sum(x, -mean_in)**2
    var_in = self.glob_mean(temp.data)

    mean_ln = mean_in.mean(1, keepdim=True)
    temp = var_in + mean_in**2
    var_ln = (temp.mean(1, keepdim=True) - mean_ln**2)

    if self.training:
      mean_bn = x.mean(0, keepdim=True)
      var_bn = x.var(0, keepdim=True)
      self.running_mean.mul_(self.momentum)
      self.running_mean.add_((1 - self.momentum) * mean_bn.data)
      self.running_var.mul_(self.momentum)
      self.running_var.add_((1 - self.momentum) * var_bn.data)
    else:
      mean_bn = torch.autograd.Variable(self.running_mean)
      var_bn = torch.autograd.Variable(self.running_var)

    softmax = nn.Softmax(dim=0)
    mean_weight = softmax(self.mean_weight)
    var_weight = softmax(self.var_weight)

    mean = mean_weight[0] * mean_in + mean_weight[1] * \
        mean_ln + mean_weight[2] * mean_bn
    var = var_weight[0] * var_in + var_weight[1] * \
        var_ln + var_weight[2] * var_bn
    inv_std = 1 / (var + self.eps).sqrt()

    x = self.glob_times(self.glob_sum2(x, -mean), inv_std)
    return x * self.weight + self.bias
