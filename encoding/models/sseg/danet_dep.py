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
from copy import deepcopy

__all__ = ['DANet_Dep', 'get_danet_dep']


class DANet_Dep(BaseNet):
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

    def __init__(self, nclass, backbone, dep_encoder = 'cnn', aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DANet_Dep, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = DANetHead(2048, nclass, norm_layer)

        # depth encoder
        self.dep_encoder = deepcopy(self.pretrained)

        self.dep_encoder = dep_encoder
        if dep_encoder == 'cnn':
            self.dep_layer0 = nn.Sequential(
                          nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                          norm_layer(64),
                          nn.ReLU(inplace=True),
                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

    def forward(self, x, dep):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        if self.dep_encoder == 'cnn':
            d0 = self.dep_layer0(dep)
            d1 = self.dep_encoder.layer1(d0)
            d2 = self.dep_encoder.layer2(d1)
            d3 = self.dep_encoder.layer3(d2)
            d = self.dep_encoder.layer4(d3)
        else:
            d = torch.nn.functional.interpolate(dep, imsize, mode='bilinear', align_corners=False)

        x = self.head(c4, d)
        x = list(x)
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        x[1] = upsample(x[1], imsize, **self._up_kwargs)
        x[2] = upsample(x[2], imsize, **self._up_kwargs)

        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])
        return tuple(outputs)


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4

        # spatial attention
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.sa = PAM_Module_Dep(inter_channels)

        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        # preprocess for depth
        self.conv5a_d = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                      norm_layer(inter_channels),
                                      nn.ReLU())
        # spatial attention output
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        # channel attention
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sc = CAM_Module(inter_channels)

        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        # channel attention output
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        # overall output
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x, dep):
        # spatial attention
        feat1 = self.conv5a(x)
        dep = self.conv5a_d(dep)

        sa_feat = self.sa(feat1, dep)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        # channel attention
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv
        sasc_output = self.conv8(feat_sum)

        output = [sasc_output, sa_output, sc_output]
        return tuple(output)


def get_danet_dep(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                  root='../../encoding/models/pretrain', **kwargs):
    r"""DANet model from the paper `"Dual Attention Network for Scene Segmentation"
    <https://arxiv.org/abs/1809.02983.pdf>`
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
    model = DANet_Dep(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model


class PAM_Module_Dep(Module):
    """ Position attention module"""
    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module_Dep, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)

        self.query_d_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_d_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)

        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x, dep):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        query = torch.cat((self.query_conv(x), self.query_d_conv(dep)), 1)            # [B, 128, h, w]
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)       # [B, hw, 128]
        key = torch.cat((self.key_conv(x), self.key_d_conv(dep)), 1)                  # [B, 128, h, w]
        proj_key = key.view(m_batchsize, -1, width*height)                            # [B, 128, hw]

        energy = torch.bmm(proj_query, proj_key)                                              # [B, hw, hw]
        attention = self.softmax(energy)                                                      # [B, hw, hw]

        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)                   # [B, 512, hw]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                               # [B, 512, hw]
        out = out.view(m_batchsize, C, height, width)                                         # [B, 512, h, w]

        out = self.gamma*out + x
        return out


