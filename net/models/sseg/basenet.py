
from addict import Dict

import torch.nn as nn
import torch.nn.functional as F

from ..backbone import get_resnet18
from ...nn import Fuse_Block, Centerpiece, Decoder

__all__ = ['BaseNet', 'get_basenet']

class BaseNet(nn.Module):

    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, root='./encoding/models/pretrain'):
        super(BaseNet, self).__init__()

        config = Dict({'ef': False, 'decoder': 'base', 'n_features': 128, 'rgbd_fuse': 'fuse',
                       'sep_fuse': False, 'cp': 'none'})
        
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
        #  - simple & pdlc
        fuse_args = {}
        #  - fuse
        fuse_args = {'out_method': 'add', 'pre_module': 'pca', 'mode': 'late',
                     'use_lamb': True, 'refine_dep': False}
        #  - gau
        # fuse_args = {'refine_rgb': True, 'refine_dep': True, 'gau_args': {'use_lamb': True}}
        #  - gf
        # fuse_args = {'info': 'psp(n); gau(b)'}
        for i in range(len(fuse_feats)):
            self.add_module('fuse%d' % i, Fuse_Block(fuse_feats[i], config.rgbd_fuse, fuse_args))
        config.fuse_args = fuse_args

        # Centerpiece
        cp_feat = 'l' if config.sep_fuse else 'f'
        # cp_args = {'size': (1, 3, 5, 7), 'up_mode': 'bilinear', 'keep_feats': False} if config.cp != 'none' else {}
        cp_args = {'size': (1, 3, 5, 7), 'up_mode': 'nearest', 'keep_feats': False} if config.cp != 'none' else {}
        self.cp = Centerpiece(config.cp, cp_feat, cp_args)
        config.cp_args = cp_args

        # Decoder
        #  - Without cp
        decoder_feat = {'level': [256, 128, 64], 'final': config.n_features}
        #  - With cp
        # decoder_feat = {'level': [512, 256, 128], 'final': config.n_features}
    
        #  - Base Net
        decoder_args = {'conv_module': 'rbb', 'level_fuse': 'add', 'feats': 'f', 'rf_conv': (False, False)}
        #  - Refine Net & GF Net
        # decoder_args = {'feats': cp_feat}
        
        self.decoder = Decoder(decoder_feat, n_classes, config.decoder, decoder_args)
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

def get_basenet(dataset='nyud', backbone='resnet18', pretrained=True, root='./encoding/models/pretrain/'):
    from ...datasets import datasets
    model = BaseNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, pretrained=pretrained, root=root)
    return model