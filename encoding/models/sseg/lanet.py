# local attention network

from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, Parameter, Softmax
from torch.nn.functional import upsample, normalize
from .base import BaseNet
from copy import deepcopy

__all__ = ['get_lanet', 'LANet']


def get_lanet(dataset='pascal_voc', backbone='resnet50', pretrained=False, dep_main=True, dep_att=False,
              dep_encode='dep',  depth_order=1, train_a=False,  root='../../encoding/models/pretrain', **kwargs):
    """
    dep_main: True/False dep input in the backbone network
    dep_att : True/False dep contribute in attention module
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
    model = LANet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, dep_main=dep_main, dep_encode=dep_encode,
                  dep_att=dep_att, depth_order=depth_order, train_a=train_a, root=root, **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model


class LANet(BaseNet):
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

    """
    def __init__(self, nclass, backbone, dep_main=False, dep_encode='dep', dep_att=False, depth_order=1, train_a=False,
                 aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(LANet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, dep_main=dep_main, **kwargs)
        self.head = LANetHead(2048, nclass, norm_layer, dep_att, depth_order, train_a)
        self.dep_main = dep_main
        self.dep_encode = 'dep'

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
        x = upsample(x, imsize, **self._up_kwargs)
        outputs = [x]
        return tuple(outputs)


class LANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, dep_att, depth_order, train_a):
        """ out_channels: nclass
        dep_att: True/False dep contribute in attention weight
        dep_order: 1/2 depth similarity based on L1 or L2
        train_a: depth similarity's weight trainable or not
        """
        super(LANetHead, self).__init__()
        inter_channels = in_channels // 4

        # spatial attention
        self.conv_s0 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU())                                             # 2048->512 reduce dimension

        # to calculate attention
        self.sa = RGBDSimilarity(inter_channels, dep_att, depth_order)
        self.gp = GeometryPrior(k=7, channels=1, multiplier=10)
        self.softmax = Softmax(dim=-1)

        # to calculate value
        self.value_conv = Conv2d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=1)   # 512->512
        self.value_unfold = nn.Unfold(kernel_size=(7, 7), padding=(3,3))

        # summarize to get output
        self.gamma = Parameter(torch.zeros(1))
        self.conv_s1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU())

        # spatial attention output
        self.conv_out = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x, d):
        # spatial attention
        m_batchsize, C, ht, wt = x.size()
        feat1 = self.conv_s0(x)                             # [B, 512, h, w]  h=w=60

        # rgbd similarity
        rgb_similarity, dep_similarity = self.sa(feat1, d)  # [B, hw, 1, 7*7]

        # to check the output
        self.rgb_similarity = rgb_similarity

        # non_zero indices. for the elements at the edge need to exclude the padded zero
        indices = self.rgb_similarity[0, :, 0, :].clone()
        indices[indices != 0] = 1                                 # [hw, 7*7]
        indices = indices.to(torch.float16).view(1, -1, 1, 7*7)   # [1, hw, 1, 7*7]

        # geometry prior
        geo_prior = self.gp()                               # [1, 1, 1, 7*7]
        geo_prior = indices*geo_prior                       # [1, hw, 1, 7*7]  mask padded zeros

        # combine RGBD similarity and geometry prior to get attention
        attention = self.softmax(rgb_similarity+geo_prior).permute(0, 1, 3, 2).contiguous()  # [B, hw, 1, 7*7] -> [B, hw, 7*7, 1]

        value = self.value_conv(feat1)                      # [B, C, h, w] with C=512
        value = self.value_unfold(value).permute(0, 2, 1).contiguous()   # [B, C*7*7, hw] -> [B, hw, C*7*7]
        value = value.view(m_batchsize, ht*wt, -1, 7*7)     # [B, hw, C, 7*7]

        sa_feat = torch.matmul(value, attention).squeeze(dim=3)           # [B, hw, C, 1] -> [B, hw, C]
        sa_feat = sa_feat.permute(0, 2, 1).contiguous().view(m_batchsize, -1, ht, wt)  # [B, C, hw] -> [B, C, h, w]

        sa_out = self.gamma * sa_feat + feat1            # [B, 512, h, w]
        sa_out = self.conv_s1(sa_out)                    # [B, 512, h, w]
        sa_output = self.conv_out(sa_out)                # [B, 40, h, w]

        return sa_output


class RGBDSimilarity(Module):
    """ Position attention module"""
    def __init__(self, in_dim, dep_att=False, depth_order=1,):
        """ dep_att: 'False' only consider RGB/RGB-D feature to calculate similarity, 'True' Consider RGB and dep
            fuse_type: 'a' add rgb_similarity and depth_similarity, 'm' multiply rgb_weight and depth_weight
            train_a:  'False' -- relative depth * constant, 'True'  -- relative depth * trainable weight
        """
        super(RGBDSimilarity, self).__init__()
        self.channel_in = in_dim
        self.dep_att = dep_att
        self.depth_order = depth_order

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)    # reduce dimension
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)    # reduce dimension
        self.key_unfold = nn.Unfold(kernel_size=(7, 7), padding=(3,3))

        self.d_unfold = nn.Unfold(kernel_size=(7,7), padding=(3,3))

    def forward(self, x, d):
        """
            inputs :
                x : input feature maps( B X C X H X W)
                d : depth feature (B X 1 X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()

        # RGB energy
        query = self.query_conv(x)                                                    # [B, 64, h, w]
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1).contiguous()       # [B, hw, 64]
        proj_query = proj_query.view(m_batchsize, width*height, 1, -1)                # [B, hw, 1, 64]
        key = self.key_conv(x)                                                        # [B, 64, h, w]
        key = self.key_unfold(key).permute(0, 2, 1).contiguous()                                  # [B, hw, 64*7*7]
        key = key.view(m_batchsize, width*height, -1, 7*7)                            # [B, hw, 64, 7*7]

        rgb_energy = torch.matmul(proj_query, key)                                    # [B, hw, 1, 7*7]

        # Depth energy
        dep_energy = None
        if self.dep_att:
            d_query = d.view(m_batchsize, 1, width*height).permute(0, 2, 1).contiguous()              # [B, hw, 1]
            d_key = self.d_unfold(d).permute(0, 2, 1).contiguous()                                     # [B, 7*7, hw] -> [B, hw, 7*7]
            dep_energy = torch.abs(d_query-d_key).view(m_batchsize, width*height, 1, -1)  # [B, hw, 7*7] -> [B, hw, 1, 7*7]

        return rgb_energy, dep_energy                                                     # [B, hw, 1, 7*7]


class GeometryPrior(torch.nn.Module):
    def __init__(self, k, channels, multiplier=10):
        super(GeometryPrior, self).__init__()
        self.channels = channels
        self.k = k
        self.l1 = torch.nn.Conv2d(2, int(multiplier * channels), 1)
        self.l2 = torch.nn.Conv2d(int(multiplier * channels), channels, 1)

    def forward(self):
        # as the paper does not infer how to construct a [2,k,k] position matrix
        # we assume that it's a kxk matrix for delta-x,and a kxk matrix for delta-y.
        # that is, [[[-1,0,1],[-1,0,1],[-1,0,1]],[[1,1,1],[0,0,0],[-1,-1,-1]]] for kernel = 3
        a_range = torch.arange(-1 * (self.k // 2), (self.k // 2) + 1).view(1, -1)      # [1, k]
        x_position = a_range.expand(self.k, a_range.shape[1])                          # [k, k]
        b_range = torch.arange((self.k // 2), -1 * (self.k // 2) - 1, -1).view(-1, 1)  # [k, 1]
        y_position = b_range.expand(b_range.shape[0], self.k)                          # [k, k]
        position = torch.cat((x_position.unsqueeze(0), y_position.unsqueeze(0)), 0).unsqueeze(0).float()  # [1, 2, 7, 7]
        if torch.cuda.is_available():
            position = position.cuda()
        out = self.l2(torch.nn.functional.relu(self.l1(position)))     # [1, 1, 7*7]
        return out.view(1, self.channels, 1, self.k ** 2)              # [1, 1, 1, 7*7]
