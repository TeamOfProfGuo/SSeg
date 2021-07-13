
from addict import Dict

import torch.nn as nn
import torch.nn.functional as F
from ..backbone import get_resnet18
from ...nn import Simple_RGBD_Fuse, RGBD_Fuse_Block, PPE_Block, Decoder

__all__ = ['BaseNet', 'get_basenet']

class BaseNet(nn.Module):

    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, root='./encoding/models/pretrain',
                 module_setting={}):
        super(BaseNet, self).__init__()

        self.early_fusion = module_setting.ef if module_setting.ef is not None else False
        decoder = module_setting.decoder if module_setting.decoder is not None else 'base'
        n_features = module_setting.n_features if module_setting.n_features is not None else 256
        rgbd_fuse = module_setting.rgbd_fuse if module_setting.rgbd_fuse is not None else 'add'
        rgbd_mode = module_setting.rgbd_mode if module_setting.rgbd_mode is not None else 'late'
        pre_module = module_setting.pre_module if module_setting.pre_module is not None else 'se'
        

        self.rgb_base = get_resnet18(input_dim=4) if self.early_fusion else get_resnet18(input_dim=3)
        self.dep_base = get_resnet18(input_dim=1)
                
        self.rgb_layer0 = nn.Sequential(self.rgb_base.conv1, self.rgb_base.bn1, self.rgb_base.relu)  # [B, 64, h/2, w/2]
        self.rgb_inpool = self.rgb_base.maxpool # [B, 64, h/4, w/4]
        self.rgb_layer1 = self.rgb_base.layer1  # [B, 64, h/4, w/4]
        self.rgb_layer2 = self.rgb_base.layer2  # [B, 128, h/8, w/8]
        self.rgb_layer3 = self.rgb_base.layer3  # [B, 256, h/16, w/16]
        self.rgb_layer4 = self.rgb_base.layer4  # [B, 512, h/32, w/32]

        self.dep_layer0 = nn.Sequential(self.dep_base.conv1, self.dep_base.bn1, self.dep_base.relu)  # [B, 64, h/2, w/2]
        self.dep_inpool = self.dep_base.maxpool # [B, 64, h/4, w/4]
        self.dep_layer1 = self.dep_base.layer1  # [B, 64, h/4, w/4]
        self.dep_layer2 = self.dep_base.layer2  # [B, 128, h/8, w/8]
        self.dep_layer3 = self.dep_base.layer3  # [B, 256, h/16, w/16]
        self.dep_layer4 = self.dep_base.layer4  # [B, 512, h/32, w/32]

        # # Simple Fuse
        # self.fuse0 = Simple_RGBD_Fuse(64)
        # self.fuse1 = Simple_RGBD_Fuse(64)
        # self.fuse2 = Simple_RGBD_Fuse(128)
        # self.fuse3 = Simple_RGBD_Fuse(256)
        # self.fuse4 = Simple_RGBD_Fuse(512)

        # RGB-D Fuse
        # out_method: add/concat
        # pre_module: se/pp/spa/context
        # mode: early(concat => attention) / late(rgb/dep attention => concat/add)
        self.fuse0 = RGBD_Fuse_Block( 64, rgbd_fuse, pre_module, rgbd_mode)
        self.fuse1 = RGBD_Fuse_Block( 64, rgbd_fuse, pre_module, rgbd_mode)
        self.fuse2 = RGBD_Fuse_Block(128, rgbd_fuse, pre_module, rgbd_mode)
        self.fuse3 = RGBD_Fuse_Block(256, rgbd_fuse, pre_module, rgbd_mode)
        self.fuse4 = RGBD_Fuse_Block(512, rgbd_fuse, pre_module, rgbd_mode)

        # # PPE/PPCE Fuse
        # self.fuse0 = PPE_Block(64)
        # self.fuse1 = PPE_Block(64)
        # self.fuse2 = PPE_Block(128)
        # self.fuse3 = PPE_Block(256)
        # self.fuse4 = PPE_Block(512)

        self.decoder = Decoder(n_features, n_classes, decoder)

    def forward(self, x, d):
        _, _, h, w = x.size()

        # Encoder
        d  = self.dep_layer0(d)    # [B, 64, h/2, w/2]
        d1 = self.dep_inpool(d)    # [B, 64, h/4, w/4]
        d1 = self.dep_layer1(d1)   # [B, 64, h/4, w/4]
        d2 = self.dep_layer2(d1)   # [B, 128, h/8, w/8]
        d3 = self.dep_layer3(d2)   # [B, 256, h/16, w/16]
        d4 = self.dep_layer4(d3)   # [B, 512, h/32, w/32]
        
        x = self.rgb_layer0(x)     # [B, 64, h/2, w/2]
        l0 = self.fuse0(x, d)      # [B, 64, h/2, w/2]

        l1 = self.rgb_inpool(l0)   # [B, 64, h/4, w/4]
        l1 = self.rgb_layer1(l1)   # [B, 64, h/4, w/4]
        l1 = self.fuse1(l1, d1)    # [B, 64, h/4, w/4]

        l2 = self.rgb_layer2(l1)   # [B, 128, h/8, w/8]
        l2 = self.fuse2(l2, d2)    # [B, 128, h/8, w/8]

        l3 = self.rgb_layer3(l2)   # [B, 256, h/16, w/16]
        l3 = self.fuse3(l3, d3)    # [B, 256, h/16, w/16]

        l4 = self.rgb_layer4(l3)   # [B, 512, h/32, w/32]
        l4 = self.fuse4(l4, d4)    # [B, 512, h/32, w/32]

        # Decoder
        feats = Dict({'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4})
        out = self.decoder(feats)

        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out

def get_basenet(dataset='nyud', backbone='resnet18', pretrained=True, root='./encoding/models/pretrain/', module_setting={}):
    from ...datasets import datasets
    model = BaseNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, pretrained=pretrained, root=root, module_setting=module_setting)
    return model