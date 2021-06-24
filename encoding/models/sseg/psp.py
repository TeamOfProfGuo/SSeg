###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import interpolate

from .base import BaseNet
from .fcn import FCNHead
from ...nn import PyramidPooling

class PSP(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(PSP, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = PSPHead(2048, nclass, norm_layer, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c4)
        x = interpolate(x, (h,w), **self._up_kwargs)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = interpolate(auxout, (h,w), **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


class PSPD(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(PSPD, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = PSPHead_Dep(2048, nclass, norm_layer, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x, dep):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c4, dep)
        x = interpolate(x, (h,w), **self._up_kwargs)  #Upsample to input size
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = interpolate(auxout, (h,w), **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


class PSPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PSPHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(PyramidPooling(in_channels, norm_layer, up_kwargs),
                                   nn.Conv2d(in_channels * 2, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


class PSPHead_Dep(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PSPHead_Dep, self).__init__()
        inter_channels = in_channels // 4
        self._up_kwargs = up_kwargs
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)
        self.pyramidPool = PyramidPooling(in_channels, norm_layer, up_kwargs)
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels * 2+5, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x, dep):
        _, _, h, w = x.size()
        dep = F.interpolate(dep,(h, w), **self._up_kwargs) #bilinear interpolate
        feat1 = F.interpolate(self.pool1(dep), (h, w), **self._up_kwargs)
        feat2 = F.interpolate(self.pool2(dep), (h, w), **self._up_kwargs)
        feat3 = F.interpolate(self.pool3(dep), (h, w), **self._up_kwargs)
        feat4 = F.interpolate(self.pool4(dep), (h, w), **self._up_kwargs)
        dep = torch.cat((dep, feat1, feat2, feat3, feat4), 1)

        x = self.pyramidPool(x)
        x = torch.cat((x, dep), 1)
        return self.conv5(x)

def get_psp(dataset='pascal_voc', backbone='resnet50s', pretrained=False,
            root='./encoding/models/pretrain', **kwargs):
    """pretrained: controls the layers other than the backbone"""
    # infer number of classes
    from ...datasets import datasets, acronyms
    # backbone is already pretrained
    model = PSP(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    print('layers after backbone pretrained {}'.format(pretrained))
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('psp_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_pspd(dataset='pascal_voc', backbone='resnet50s', pretrained=False, root='./encoding/models/pretrain', **kwargs):
    """pretrained: controls the layers other than the backbone"""
    # infer number of classes
    from ...datasets import datasets, acronyms
    # backbone is already pretrained
    model = PSPD(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    print('layers after backbone pretrained {}'.format(pretrained))
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('psp_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_psp_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_psp_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_psp('ade20k', 'resnet50s', pretrained, root=root, **kwargs)
