###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, Parameter, Softmax
from torch.nn.functional import upsample, normalize
from ...nn import PAM_Module
from ...nn import CAM_Module
from .base import BaseNet
from ...nn import PyramidPooling
from torch.nn import functional as F

__all__ = ['DANetPSP', 'get_danet_psp']


class DANetPSP(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    """

    def __init__(self, nclass, backbone, dep_main=False, dep_psp = True, fuse_type='m', depth_order=1, train_a=False,
                 aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DANetPSP, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, dim=8, **kwargs)
        self.head = DANetHead(2048, nclass, norm_layer,)
        self.dep_psp = dep_psp
        self.dep_main = dep_main
        self.dep_psp = PSPD(self._up_kwargs)

    def forward(self, x, dep):
        imsize = x.size()[2:]
        if self.dep_main:
            if self.dep_psp:
                dep = self.dep_psp(dep)    # 5 channel
            x = torch.cat((x, dep), 1)
            _, _, c3, c4 = self.base_forward(x)
        else:
            _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = list(x)
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        x[1] = upsample(x[1], imsize, **self._up_kwargs)
        x[2] = upsample(x[2], imsize, **self._up_kwargs)

        outputs = [x[0], x[1], x[2]]
        return tuple(outputs)


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv_s0 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv_c0 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv_s1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv_c1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv_s2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv_c2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv_out = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv_s0(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv_s1(sa_feat)
        sa_output = self.conv_s2(sa_conv)

        feat2 = self.conv_c0(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv_c1(sc_feat)
        sc_output = self.conv_c2(sc_conv)

        feat_sum = sa_conv + sc_conv
        sasc_output = self.conv_out(feat_sum)

        output = [sasc_output, sa_output, sc_output]
        return tuple(output)


def get_danet_psp(dataset='pascal_voc', backbone='resnet50', dep_main=True, pretrained=False,
                  dep_psp=True, fuse_type ='m', depth_order=1, train_a=False,
                  root='../../encoding/models/pretrain', **kwargs):
    """
    fuse_type: 'a' add rgb_similarity and depth_similarity
          'm' multiply rgb_weight and depth_weight
    dep_psp: True -- process depth using PSP module
    depth_encode: dep -- resize depth feature directly
                  cnn -- use a deep cnn to process depth feature
    train_a:  False -- relative depth * constant
              True  -- relative depth * trainable weight

    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
        'cityscapes': 'cityscapes',
    }
    # infer number of classes
    from ...datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = DANetPSP(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, dep_main=dep_main, dep_psp=dep_psp,
                    fuse_type=fuse_type, depth_order=depth_order, train_a=train_a, root=root, **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model


class PSPD(nn.Module):
    def __init__(self, up_kwargs):
        super(PSPD, self).__init__()
        self._up_kwargs = up_kwargs
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.pool1(x), (h, w), **self._up_kwargs)
        feat2 = F.interpolate(self.pool2(x), (h, w), **self._up_kwargs)
        feat3 = F.interpolate(self.pool3(x), (h, w), **self._up_kwargs)
        feat4 = F.interpolate(self.pool4(x), (h, w), **self._up_kwargs)
        out = torch.cat((x, feat1, feat2, feat3, feat4), 1)
        return out