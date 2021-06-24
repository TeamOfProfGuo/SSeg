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
from torch.nn.functional import upsample, normalize
from torch.nn import Module, Sequential, Conv2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
#from ...nn import PAM_Module
from ...nn import CAM_Module
from ...nn import PyramidPooling
from .base import BaseNet

__all__ = ['PPANet', 'get_ppanet']


class PPANet(BaseNet):
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

    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(PPANet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = PPANetHead(2048, nclass, norm_layer, self._up_kwargs)

    def forward(self, x, dep):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = list(x)
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        x[1] = upsample(x[1], imsize, **self._up_kwargs)
        x[2] = upsample(x[2], imsize, **self._up_kwargs)

        outputs = [x[0], x[1], x[2]]
        return tuple(outputs)


class PPANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PPANetHead, self).__init__()
        inter_channels = in_channels // 4    # 512

        # pyramid pooling
        self.pyramid_pool = PyramidPooling(in_channels, norm_layer, up_kwargs)
        self.conv_pp = nn.Sequential(nn.Conv2d(in_channels * 2, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(True))

        # spatial attention
        self.conv_s0 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU())
        self.sa = PPA_Module(inter_channels)
        self.conv_s1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU())

        # channel attention
        self.conv_c0 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU())
        self.sc = CAM_Module(inter_channels)
        self.conv_c1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU())

        self.conv_s2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv_c2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        pp_feat = self.pyramid_pool(x)     # [4096, 60, 60]
        pp_feat = self.conv_pp(pp_feat)    # [512, 60, 60]

        feat1 = self.conv_s0(x)
        sa_feat = self.sa(feat1, pp_feat)
        sa_conv = self.conv_s1(sa_feat)

        sa_output = self.conv_s2(sa_conv)

        feat2 = self.conv_c0(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv_c1(sc_feat)
        sc_output = self.conv_c2(sc_conv)

        feat_sum = sa_conv + sc_conv
        sasc_output = self.conv_2(feat_sum)

        output = [sasc_output, sa_output, sc_output]
        return tuple(output)


class PPA_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PPA_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=-1)
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, pp_feat):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        proj_value = pp_feat.view(m_batchsize, -1, width*height)     # [B, 4096, wh]

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + pp_feat
        return out


def get_ppanet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
               root='../../encoding/models/pretrain', **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
        'cityscapes': 'cityscapes',
    }
    # infer number of classes
    from ...datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = PPANet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model

