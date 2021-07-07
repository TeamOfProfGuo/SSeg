import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# from ...nn import PAM_Module, CAM_Module
from ...nn import BaseRefineNetBlock, ResidualConvUnit, MRF_Concat_5_2, RGBD_Fuse_Block

__all__ = ['RN_rgbd_fuse', 'get_rn_rgbd_fuse']

class RN_rgbd_fuse(nn.Module):

    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, root='./encoding/models/pretrain',
                 module_setting={}):
        super(RN_rgbd_fuse, self).__init__()
        # self.do = nn.Dropout(p=0.5)

        self.early_fusion = module_setting.ef if module_setting.ef is not None else False
        self.use_crp = module_setting.crp if module_setting.crp is not None else False
        n_features = module_setting.n_features if module_setting.n_features is not None else 256
        rgbd_fuse = module_setting.rgbd_fuse if module_setting.rgbd_fuse is not None else 'add'
        pre_module = module_setting.pre_module if module_setting.pre_module is not None else 'se'
        

        self.rgb_base = get_resnet18(input_dim=4) if self.early_fusion else get_resnet18(input_dim=3)
        self.dep_base = get_resnet18(input_dim=1)
                
        self.rgb_layer0 = nn.Sequential(self.rgb_base.conv1, self.rgb_base.bn1, 
                                        self.rgb_base.relu, self.rgb_base.maxpool)  # [B, 64, h/4, w/4]
        self.rgb_layer1 = self.rgb_base.layer1  # [B, 64, h/4, w/4]
        self.rgb_layer2 = self.rgb_base.layer2  # [B, 128, h/8, w/8]
        self.rgb_layer3 = self.rgb_base.layer3  # [B, 256, h/16, w/16]
        self.rgb_layer4 = self.rgb_base.layer4  # [B, 512, h/32, w/32]

        self.dep_layer0 = nn.Sequential(self.dep_base.conv1, self.dep_base.bn1,
                                        self.dep_base.relu, self.dep_base.maxpool)  # [B, 64, h/4, w/4]
        self.dep_layer1 = self.dep_base.layer1  # [B, 64, h/4, w/4]
        self.dep_layer2 = self.dep_base.layer2  # [B, 128, h/8, w/8]
        self.dep_layer3 = self.dep_base.layer3  # [B, 256, h/16, w/16]
        self.dep_layer4 = self.dep_base.layer4  # [B, 512, h/32, w/32]

        self.fuse1 = RGBD_Fuse_Block( 64, out_method=rgbd_fuse, pre_module=pre_module)
        self.fuse2 = RGBD_Fuse_Block(128, out_method=rgbd_fuse, pre_module=pre_module)
        self.fuse3 = RGBD_Fuse_Block(256, out_method=rgbd_fuse, pre_module=pre_module)
        self.fuse4 = RGBD_Fuse_Block(512, out_method=rgbd_fuse, pre_module=pre_module)

        self.refine_conv1 = nn.Conv2d( 64, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.refine_conv2 = nn.Conv2d(128, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.refine_conv3 = nn.Conv2d(256, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.refine_conv4 = nn.Conv2d(512, 2*n_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.refine4 = RefineNetBlock(2*n_features, self.use_crp, (2*n_features, 32))
        self.refine3 = RefineNetBlock(n_features, self.use_crp, (2*n_features, 32), (n_features, 16))
        self.refine2 = RefineNetBlock(n_features, self.use_crp, (n_features, 16), (n_features, 8))
        self.refine1 = RefineNetBlock(n_features, self.use_crp, (n_features, 8), (n_features, 4))

        self.out_conv = nn.Sequential(
            ResidualConvUnit(n_features), ResidualConvUnit(n_features),
            nn.Conv2d(n_features, n_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x, d):
        _, _, h, w = x.size()

        d = self.dep_layer0(d)
        d1 = self.dep_layer1(d)    # [B, 64, h/4, w/4]
        d2 = self.dep_layer2(d1)   # [B, 128, h/8, w/8]
        d3 = self.dep_layer3(d2)   # [B, 256, h/16, w/16]
        d4 = self.dep_layer4(d3)   # [B, 512, h/32, w/32]
        
        x = self.rgb_layer0(x)
        l1 = self.rgb_layer1(x)    # [B, 64, h/4, w/4]
        l1 = self.fuse1(l1, d1)
        l2 = self.rgb_layer2(l1)   # [B, 128, h/8, w/8]
        l2 = self.fuse2(l2, d2)
        l3 = self.rgb_layer3(l2)   # [B, 256, h/16, w/16]
        l3 = self.fuse3(l3, d3)
        l4 = self.rgb_layer4(l3)   # [B, 512, h/32, w/32]
        l4 = self.fuse4(l4, d4)

        l1 = self.refine_conv1(l1)
        l2 = self.refine_conv2(l2)
        l3 = self.refine_conv3(l3)
        l4 = self.refine_conv4(l4)

        path4 = self.refine4(l4)          # [B, 512, h/32, w/32]
        path3 = self.refine3(path4, l3)   # [B, 256, h/16, w/16]
        path2 = self.refine2(path3, l2)   # [B, 256, h/8, w/8]
        path1 = self.refine1(path2, l1)   # [B, 256, h/4, w/4]
        out = self.out_conv(path1)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)

        # danet_out = self.danet(l4)
        # out = F.interpolate(danet_out[0], (h, w), mode='bilinear', align_corners=True)
        
        return out

def get_rn_rgbd_fuse(dataset='nyud', backbone='resnet18', pretrained=True, root='./encoding/models/pretrain/', module_setting={}):
    from ...datasets import datasets
    model = RN_rgbd_fuse(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, pretrained=pretrained, root=root, module_setting=module_setting)
    return model

def get_resnet18(pretrained=True, input_dim = 3, f_path='./encoding/models/pretrain/resnet18-5c106cde.pth'):
    assert input_dim in (1, 3, 4)
    model = models.resnet18(pretrained=False)

    if pretrained:
        # Check weights file
        if not os.path.exists(f_path):
            raise FileNotFoundError('The pretrained model cannot be found.')
        
        if input_dim != 3:
            model.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            weights = torch.load(f_path)
            for k, v in weights.items():
                weights[k] = v.data
            conv1_ori = weights['conv1.weight']
            conv1_new = torch.zeros((64, input_dim, 7, 7), dtype=torch.float32)
            if input_dim == 4:
                conv1_new[:, :3, :, :] = conv1_ori
                conv1_new[:,  3, :, :] = conv1_ori[:,  1, :, :]
            else:
                conv1_new[:,  0, :, :] = conv1_ori[:,  1, :, :]
            weights['conv1.weight'] = conv1_new
            model.load_state_dict(weights, strict=False)
        else:
            model.load_state_dict(torch.load(f_path), strict=False)
    else:
        raise ValueError('Please use pretrained resnet18.')
    
    return model

class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, use_crp, *shapes):
        super().__init__(features, use_crp, ResidualConvUnit, MRF_Concat_5_2, nn.Identity, *shapes)