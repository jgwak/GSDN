import numpy as np
import torch
import torch.nn as nn

from lib.pipelines.base import BasePipeline
import lib.utils as utils
import models


class Segmentation(BasePipeline):

  TARGET_METRIC = 'mIoU'

  def get_metric(self, val_dict):
    return np.nanmean(val_dict['semantic_iou'])

  def initialize_hists(self):
    return {
        'semantic_hist': np.zeros((self.num_labels, self.num_labels)),
    }

  def __init__(self, config, dataset):
    super().__init__(config, dataset)

    backbone_model_class = models.load_model(config.backbone_model)
    self.backbone = backbone_model_class(dataset.NUM_IN_CHANNEL, config).to(self.device)
    self.segmentation = models.segmentation.SparseFeatureUpsampleNetwork(
        self.backbone.out_channels, self.backbone.OUT_PIXEL_DIST, dataset.NUM_LABELS,
        config).to(self.device)
    self.criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)
    self.num_labels = dataset.NUM_LABELS

  def forward(self, datum, is_train):
    backbone_outputs = self.backbone(datum['sinput'])
    outputs = self.segmentation(backbone_outputs)
    outcoords, outfeats = outputs.decomposed_coordinates_and_features
    assert torch.allclose(datum['coords'][:, 1:], outputs.C[:, 1:])
    pred = torch.argmax(outputs.F, 1)
    return {'outputs': outputs, 'pred': pred}

  @staticmethod
  def update_meters(meters, hists, loss_dict):
    for k, v in loss_dict.items():
      if k == 'semantic_hist':
        assert 'semantic_hist' in hists
        hists['semantic_hist'] += v
        meters['semantic_iou'] = utils.per_class_iu(hists['semantic_hist'])
      else:
        meters[k].update(v)
    return meters, hists

  def evaluate(self, datum, output):
    return {
        'semantic_hist': utils.fast_hist(
            output['pred'].cpu().numpy(), datum['semantic_target'].cpu().numpy(), self.num_labels)
    }

  def loss(self, datum, output):
    score = utils.precision_at_one(output['pred'], datum['semantic_target'])
    return {
        'score': torch.FloatTensor([score])[0].to(output['outputs'].F),
        'loss': self.criterion(output['outputs'].F, datum['semantic_target'])
    }
