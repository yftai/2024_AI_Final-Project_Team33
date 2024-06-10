from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels


def get_model(model_name='resnet18', **kwargs):
    pretrained = 'imagenet'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000,
                                                  pretrained=pretrained)

    if 'resnet' in model_name:
        model.avgpool = nn.AdaptiveAvgPool2d(1)
    else:
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, 1)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    return model
