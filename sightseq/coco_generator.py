# Copyright (c) 2019-present, Zhiqiang Wang.

import torch


class ObjectDetectionGenerator(object):
    """Scores the target for a given source image."""

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of images with best path decoding."""
        assert len(models) == 1

        model = ObjectDetectionEnsembleModel(models)
        model.eval()
        net_input = sample['image']
        hypos = model.forward_featurize(net_input)

        return hypos


class ObjectDetectionEnsembleModel(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    @torch.no_grad()
    def forward_featurize(self, net_input):
        return self._featurize_one(self.models[0], net_input)

    def _featurize_one(self, model, net_input):
        net_output = model(net_input)
        hypos = model.get_box_detections(net_output)

        return hypos
