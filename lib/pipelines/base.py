import torch
import torch.nn as nn
import MinkowskiEngine as ME

import lib.utils as utils
import lib.solvers as solvers


class BasePipeline(nn.Module):

  def __init__(self, config, dataset):
    nn.Module.__init__(self)

    self.config = config
    self.device = utils.get_torch_device(config.is_cuda)
    self.num_labels = dataset.NUM_LABELS
    self.dataset = dataset

  def initialize_optimizer(self, config):
    return {
        'default': solvers.initialize_optimizer(self.parameters(), config)
    }

  def initialize_scheduler(self, optimizers, config, last_step=-1):
    schedulers = {}
    for key, optimizer in optimizers.items():
      schedulers[key] = solvers.initialize_scheduler(optimizer, config, last_step=last_step)
    return schedulers

  def load_optimizer(self, optimizers, state_dict):
    if set(optimizers) == set(state_dict):
      for key in optimizers:
        optimizers[key].load_state_dict(state_dict[key])
    elif 'param_groups' in state_dict:
      optimizers['default'].load_state_dict(state_dict)
    else:
      raise ValueError('Unknown optimizer parameter format.')

  def reset_gradient(self, optimizers):
    for optimizer in optimizers.values():
      optimizer.zero_grad()

  def step_optimizer(self, output, optimizers, schedulers, iteration):
    assert set(optimizers) == set(schedulers)
    for key in optimizers:
      optimizers[key].step()
      schedulers[key].step()

  @staticmethod
  def _convert_target2si(target, ignore_label):
    return target[:, 0].long(), target[:, 1].long()

  def initialize_hists(self):
    return dict()

  @staticmethod
  def update_meters(meters, hists, loss_dict):
    for k, v in loss_dict.items():
      if k == 'ap':
        for histk in hists:
          if histk.startswith('ap_'):
            hists[histk].step(v['pred'], v['gt'])
            meters[histk] = hists[histk]
      else:
        meters[k].update(v)
    return meters, hists

  def load_datum(self, data_iter, has_gt=True):
    datum = data_iter.next()

    # Preprocess input
    if self.dataset.USE_RGB and self.config.normalize_color:
      datum['input'][:, :3] = datum['input'][:, :3] / 255. - 0.5
    datum['sinput'] = ME.SparseTensor(datum['input'], datum['coords']).to(self.device)

    # Preprocess target
    if has_gt:
      if 'rpn_bbox' in datum:
        datum['rpn_bbox'] = torch.from_numpy(datum['rpn_bbox']).float().to(self.device)
      if 'rpn_rotation' in datum and datum['rpn_rotation'] is not None:
        datum['rpn_rotation'] = torch.from_numpy(datum['rpn_rotation']).float().to(self.device)
      if 'rpn_match' in datum:
        datum['rpn_match'] = torch.from_numpy(datum['rpn_match']).to(self.device)
      datum['target'] = datum['target'].to(self.device)
      semantic_target, instance_target = self._convert_target2si(datum['target'],
                                                                 self.config.ignore_label)
      datum.update({
          'semantic_target': semantic_target.to(self.device),
          'instance_target': instance_target.to(self.device),
      })

    return datum

  def evaluate(self, datum, output):
    return {}
