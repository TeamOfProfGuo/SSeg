
from addict import Dict

import torch.nn as nn
import torch.nn.functional as F

from ..backbone import get_resnet18
from ...nn import Fuse_Block, Centerpiece, Decoder

__all__ = ['Prl_BaseNet', 'get_prl_basenet']

class Prl_BaseNet(nn.Module):

    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, root='./encoding/models/pretrain', setting={}):
        super().__init__()

        setting = Dict(setting)
        config = setting.general
        
        self.sep_fuse = config.sep_fuse

        # Get backbone
        self.rgb_base = get_resnet18(input_dim=4) if config.early_fusion else get_resnet18(input_dim=3)
        self.dep_base = get_resnet18(input_dim=1)
        
        # Divide backbone into 5 parts
        self.rgb_layer0 = nn.Sequential(self.rgb_base.conv1,
                                        self.rgb_base.bn1,
                                        self.rgb_base.relu)  # [B, 64, h/2, w/2]
        self.dep_layer0 = nn.Sequential(self.dep_base.conv1, 
                                        self.dep_base.bn1,
                                        self.dep_base.relu)  # [B, 64, h/2, w/2]
        self.rgb_inpool = self.rgb_base.maxpool              # [B, 64, h/4, w/4]
        self.dep_inpool = self.dep_base.maxpool              # [B, 64, h/4, w/4]
        for i in range(1, 5):
            self.add_module('rgb_layer%d' % i, self.rgb_base.__getattr__('layer%d' % i))
            self.add_module('dep_layer%d' % i, self.dep_base.__getattr__('layer%d' % i))

        # Fuse Block
        fuse_feats = [64, 64, 128, 256, 512]
        fuse_args = setting.fuse_args
        for i in range(len(fuse_feats)):
            self.add_module('fuse%d' % i, Fuse_Block(fuse_feats[i], config.rgbd_fuse, fuse_args))
        config.fuse_args = fuse_args

        # Centerpiece
        cp_feat = 'l' if config.sep_fuse else 'f'
        cp_args = setting.cp_args if config.cp != 'none' else {}
        self.cp = Centerpiece(config.cp, cp_feat, cp_args)
        config.cp_args = cp_args

        # Decoder feats
        decoder_feat = setting.decoder_feat
        decoder_args = setting.decoder_args        
        self.decoder = Decoder(decoder_feat, n_classes, config.decoder, decoder_args)
        config.decoder_feat = decoder_feat
        config.decoder_args = decoder_args

        self.config = config

    def forward(self, x, d):
        _, _, h, w = x.size()

        # Encoder
        l0 = self.rgb_layer0(x)         # [B, 64, h/2, w/2]
        d0 = self.dep_layer0(d)         # [B, 64, h/2, w/2]
        f0, d0 = self.fuse0(l0, d0)     # [B, 64, h/2, w/2]

        l1 = self.rgb_inpool(l0 if self.sep_fuse else f0)
        l1 = self.rgb_layer1(l1)       # [B, 64, h/4, w/4]
        d1 = self.dep_inpool(d0)       # [B, 64, h/4, w/4]
        d1 = self.dep_layer1(d1)       # [B, 64, h/4, w/4]
        f1, d1 = self.fuse1(l1, d1)    # [B, 64, h/4, w/4]

        l2 = self.rgb_layer2(l1 if self.sep_fuse else f1)
        d2 = self.dep_layer2(d1)       # [B, 128, h/8, w/8]
        f2, d2 = self.fuse2(l2, d2)    # [B, 128, h/8, w/8]

        l3 = self.rgb_layer3(l2 if self.sep_fuse else f2)
        d3 = self.dep_layer3(d2)       # [B, 256, h/16, w/16]
        f3, d3 = self.fuse3(l3, d3)    # [B, 256, h/16, w/16]

        l4 = self.rgb_layer4(l3 if self.sep_fuse else f3)
        d4 = self.dep_layer4(d3)       # [B, 512, h/32, w/32]
        f4, d4 = self.fuse4(l4, d4)    # [B, 512, h/32, w/32]

        feats = Dict({'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4,
                      'd1': d1, 'd2': d2, 'd3': d3, 'd4': d4,
                      'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4})
        
        # Centerpiece
        feats = self.cp(feats)

        # Decoder
        out = self.decoder(feats)

        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out

def get_prl_basenet(dataset='nyud', backbone='resnet18', pretrained=True, root='./encoding/models/pretrain/', config={}):
    from ...datasets import datasets
    model = Prl_BaseNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, pretrained=pretrained, root=root, setting=config)
    return model