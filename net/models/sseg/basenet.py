
from addict import Dict

import torch.nn as nn
import torch.nn.functional as F
from ..backbone import get_resnet18
from ...nn import Simple_RGBD_Fuse, RGBD_Fuse_Block, PPE_Block, GAU_Fuse, Decoder

__all__ = ['BaseNet', 'get_basenet']

class BaseNet(nn.Module):

    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, root='./encoding/models/pretrain',
                 config_setting={}):
        super(BaseNet, self).__init__()

        # Update config
        config = Dict({'ef': False, 'decoder': 'base', 'n_features': 256, 'rgbd_op': 'add',
                       'rgbd_fuse': 'fuse', 'rgbd_mode': 'late', 'pre_module': 'se'})
        for k, v in config_setting.items():
            config[k] = v
        
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

        # Fuse modules
        fuse_dict = {'simple': Simple_RGBD_Fuse, 'fuse': RGBD_Fuse_Block, 'gau': GAU_Fuse}
        # fuse_args = {'out_method': config.rgbd_op, 'pre_module': config.pre_module, 'mode': config.rgbd_mode}
        fuse_args = {}

        self.fuse0 = fuse_dict[config.rgbd_fuse]( 64, **fuse_args)
        self.fuse1 = fuse_dict[config.rgbd_fuse]( 64, **fuse_args)
        self.fuse2 = fuse_dict[config.rgbd_fuse](128, **fuse_args)
        self.fuse3 = fuse_dict[config.rgbd_fuse](256, **fuse_args)
        self.fuse4 = fuse_dict[config.rgbd_fuse](512, **fuse_args)

        # Decoder
        self.decoder = Decoder(config.n_features, n_classes, config.decoder)

    def forward(self, x, d):
        _, _, h, w = x.size()

        # Encoder
        x = self.rgb_layer0(x)         # [B, 64, h/2, w/2]
        d = self.dep_layer0(d)         # [B, 64, h/2, w/2]
        x, d = self.fuse0(x, d)        # [B, 64, h/2, w/2]

        l1 = self.rgb_inpool(x)        # [B, 64, h/4, w/4]
        l1 = self.rgb_layer1(l1)       # [B, 64, h/4, w/4]
        d1 = self.dep_inpool(d)        # [B, 64, h/4, w/4]
        d1 = self.dep_layer1(d1)       # [B, 64, h/4, w/4]
        l1, d1 = self.fuse1(l1, d1)    # [B, 64, h/4, w/4]

        l2 = self.rgb_layer2(l1)       # [B, 128, h/8, w/8]
        d2 = self.dep_layer2(d1)       # [B, 128, h/8, w/8]
        l2, d2 = self.fuse2(l2, d2)    # [B, 128, h/8, w/8]

        l3 = self.rgb_layer3(l2)       # [B, 256, h/16, w/16]
        d3 = self.dep_layer3(d2)       # [B, 256, h/16, w/16]
        l3, d3 = self.fuse3(l3, d3)    # [B, 256, h/16, w/16]

        l4 = self.rgb_layer4(l3)       # [B, 512, h/32, w/32]
        d4 = self.dep_layer4(d3)       # [B, 512, h/32, w/32]
        l4, d4 = self.fuse4(l4, d4)    # [B, 512, h/32, w/32]

        # Decoder
        feats = Dict({'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4,
                      'd1': d1, 'd2': d2, 'd3': d3, 'd4': d4})
        out = self.decoder(feats)

        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out

def get_basenet(dataset='nyud', backbone='resnet18', pretrained=True, root='./encoding/models/pretrain/', config={}):
    from ...datasets import datasets
    model = BaseNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, pretrained=pretrained, root=root, config_setting=config)
    return model