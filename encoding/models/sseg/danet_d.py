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
#from ...nn import PAM_Module
from ...nn import CAM_Module
from .base import BaseNet
from copy import deepcopy

__all__ = ['DANet_D', 'get_danet_d']


class DANet_D(BaseNet):
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

    def __init__(self, nclass, backbone, dep_main=False, dep_encode='dep', fuse_type='m', depth_order=1, train_a=False, post_conv=True,
                 aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        dim = 4 if dep_main else 3
        super(DANet_D, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, dim=dim, **kwargs)
        self.head = DANetHead(2048, nclass, norm_layer, fuse_type, depth_order, train_a, post_conv)
        self.dep_encode = dep_encode
        self.dep_main = dep_main

        # depth encoder
        if self.dep_encode == 'cnn':
            self.dep_encoder = deepcopy(self.pretrained)
            self.dep_layer0 = nn.Sequential(
                          nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                          norm_layer(64),
                          nn.ReLU(inplace=True),
                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

    def forward(self, x, dep):
        imsize = x.size()[2:]
        if self.dep_main:
            x = torch.cat((x, dep), 1)
            _, _, c3, c4 = self.base_forward(x)
        else:
            _, _, c3, c4 = self.base_forward(x)

        if self.dep_encode == 'cnn':
            d0 = self.dep_layer0(dep)
            d1 = self.dep_encoder.layer1(d0)
            d2 = self.dep_encoder.layer2(d1)
            d3 = self.dep_encoder.layer3(d2)
            d = self.dep_encoder.layer4(d3)
        else:
            d = torch.nn.functional.interpolate(dep, c4.size()[2:], mode='bilinear', align_corners=False)

        x = self.head(c4, d)
        x = list(x)
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        x[1] = upsample(x[1], imsize, **self._up_kwargs)
        x[2] = upsample(x[2], imsize, **self._up_kwargs)

        outputs = [x[0], x[1], x[2]]
        return tuple(outputs)


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, fuse_type, depth_order, train_a, post_conv):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.post_conv = post_conv

        # spatial attention
        self.conv_s0 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.sa = PAM_Module_Dep(inter_channels, fuse_type, depth_order, train_a)

        self.conv_s1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        # spatial attention output
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        # channel attention
        self.conv_c0 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sc = CAM_Module(inter_channels)

        self.conv_c1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        # channel attention output
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        # overall output
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x, d):
        # spatial attention
        feat1 = self.conv_s0(x)
        sa_conv = self.sa(feat1, d)
        if self.post_conv:
            sa_conv = self.conv_s1(sa_conv)
        #else:
        #    print('== there is no conv after PAM module==')
        sa_output = self.conv6(sa_conv)

        # channel attention
        feat2 = self.conv_c0(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv_c1(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv
        sasc_output = self.conv8(feat_sum)

        output = [sasc_output, sa_output, sc_output]
        return tuple(output)


def get_danet_d(dataset='pascal_voc', backbone='resnet50', dep_main=False, pretrained=False, dep_encode='dep',
                fuse_type ='m', depth_order=1, train_a=False, post_conv=True,
                root='../../encoding/models/pretrain', **kwargs):
    """
    fuse_type: 'a' add rgb_similarity and depth_similarity
          'm' multiply rgb_weight and depth_weight
    depth_order: 1 -- calculate absolute value of relative depth
                 2 -- calculate squared relative depth
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
    model = DANet_D(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, dep_main=dep_main, dep_encode=dep_encode,
                    fuse_type=fuse_type, depth_order=depth_order, train_a=train_a, post_conv=post_conv, root=root, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model


class PAM_Module_Dep(Module):
    """ Position attention module"""
    # Ref from SAGAN
    def __init__(self, in_dim, fuse_type, depth_order, train_a):
        super(PAM_Module_Dep, self).__init__()
        self.channel_in = in_dim
        self.depth_order = depth_order
        self.fuse_type = fuse_type

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = Parameter(torch.zeros(1))
        if self.fuse_type == 'a':
            if train_a:
                self.lamb = Parameter(torch.zeros(1))
            else:
                self.lamb = 5
        elif self.fuse_type == 'm':
            if train_a:
                self.alpha = Parameter(torch.zeros(1))
                print('alpha is a trainable param')
            else:
                self.alpha = -8
                print('alpha is a constant')

        self.softmax = Softmax(dim=-1)
        self.d_softmax = Softmax(dim=-1)

    def forward(self, x, d):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()

        # RGB similarity
        query = self.query_conv(x)                                                    # [B, 64, h, w]
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)       # [B, hw, 64]
        key = self.key_conv(x)                                                        # [B, 64, h, w]
        proj_key = key.view(m_batchsize, -1, width*height)                            # [B, 64, hw]
        energy = torch.bmm(proj_query, proj_key)                                      # [B, hw, hw]

        # Depth similarity
        d1 = d.view(m_batchsize, -1, width*height)     # [B, 1, hw]
        d_query = d1.permute(0, 2, 1)                  # [B, hw, 1]
        dd = d_query - d1                              # [B, hw, hw]
        if self.depth_order == 1:
            dd = torch.abs(dd)
        elif self.depth_order == 2:
            dd = torch.pow(dd, 2)

        # combine rgb similarity and depth similarity
        if self.fuse_type == 'a':
            similarity = energy + self.lamb * dd
            attention = self.softmax(similarity)                                              # [B, hw, hw]
        elif self.fuse_type == 'm':
            rgb_attention = self.softmax(energy)                                              # [B, hw, hw]
            dep_attention = self.d_softmax(self.alpha*dd) * width*height
            attention = rgb_attention * dep_attention

        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)                   # [B, 512, hw]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                               # [B, 512, hw]
        out = out.view(m_batchsize, C, height, width)                                         # [B, 512, h, w]

        out = self.gamma*out + x
        return out


