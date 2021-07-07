import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# from ...nn import PAM_Module, CAM_Module
from ...nn import BaseRefineNetBlock, ResidualConvUnit, MRF_Concat_5_2, RGBD_Fuse_Block

__all__ = ['BaseNet', 'get_basenet']

class BaseNet(nn.Module):

    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, root='./encoding/models/pretrain',
                 module_setting={}):
        super(BaseNet, self).__init__()
        # self.do = nn.Dropout(p=0.5)

        self.decoder = module_setting.decoder if module_setting.decoder is not None else 'base'
        self.early_fusion = module_setting.ef if module_setting.ef is not None else False
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

        # SPA Fuse
        # out_method: add/concat
        # pre_module: se/pp/spa/context
        # mode: early(concat => attention) / late(attention => concat/add)
        self.fuse0 = RGBD_Fuse_Block( 64, out_method=rgbd_fuse, pre_module=pre_module, mode=rgbd_mode)
        self.fuse1 = RGBD_Fuse_Block( 64, out_method=rgbd_fuse, pre_module=pre_module, mode=rgbd_mode)
        self.fuse2 = RGBD_Fuse_Block(128, out_method=rgbd_fuse, pre_module=pre_module, mode=rgbd_mode)
        self.fuse3 = RGBD_Fuse_Block(256, out_method=rgbd_fuse, pre_module=pre_module, mode=rgbd_mode)
        self.fuse4 = RGBD_Fuse_Block(512, out_method=rgbd_fuse, pre_module=pre_module, mode=rgbd_mode)

        if self.decoder == 'refine':
            self.refine_conv1 = nn.Conv2d( 64, n_features, kernel_size=3, stride=1, padding=1, bias=False)
            self.refine_conv2 = nn.Conv2d(128, n_features, kernel_size=3, stride=1, padding=1, bias=False)
            self.refine_conv3 = nn.Conv2d(256, n_features, kernel_size=3, stride=1, padding=1, bias=False)
            self.refine_conv4 = nn.Conv2d(512, 2*n_features, kernel_size=3, stride=1, padding=1, bias=False)

            self.refine4 = RefineNetBlock(2*n_features, (2*n_features, 32))
            self.refine3 = RefineNetBlock(n_features, (2*n_features, 32), (n_features, 16))
            self.refine2 = RefineNetBlock(n_features, (n_features, 16), (n_features, 8))
            self.refine1 = RefineNetBlock(n_features, (n_features, 8), (n_features, 4))

            self.out_conv = nn.Sequential(
                ResidualConvUnit(n_features), ResidualConvUnit(n_features),
                nn.Conv2d(n_features, n_classes, kernel_size=1, stride=1, padding=0, bias=True))
        elif self.decoder == 'base':
            # self.refine1 = Simple_Level_Fuse()
            self.refine2 = Simple_Level_Fuse(64)
            self.refine3 = Simple_Level_Fuse(128)
            self.refine4 = Simple_Level_Fuse(256)
            

            # # v1.0 RCU
            # self.up2 = nn.Sequential(ResidualConvUnit(128), LearnedUpUnit(128, 64))
            # self.up3 = nn.Sequential(ResidualConvUnit(256), LearnedUpUnit(256, 128))
            # self.up4 = nn.Sequential(ResidualConvUnit(512), LearnedUpUnit(512, 256))

            # v1.1 CBR
            self.up2 = nn.Sequential(CBR(128,  64), LearnedUpUnit(64))
            self.up3 = nn.Sequential(CBR(256, 128), LearnedUpUnit(128))
            self.up4 = nn.Sequential(CBR(512, 256), LearnedUpUnit(256))

            # # v1.2 BB
            # self.up4 = nn.Sequential(BasicBlock(512, 512), BasicBlock(512, 256, upsample=True))
            # self.up3 = nn.Sequential(BasicBlock(256, 256), BasicBlock(256, 128, upsample=True))
            # self.up2 = nn.Sequential(BasicBlock(128, 128), BasicBlock(128, 64, upsample=True))

            self.out_conv = nn.Sequential(
                # ResidualConvUnit(64), ResidualConvUnit(64),
                CBR(64, n_features), CBR(n_features, n_features),
                # BasicBlock(64, 128, upsample=True), # BasicBlock(128, 128),
                nn.Conv2d(n_features, n_classes, kernel_size=1, stride=1, padding=0, bias=True))
        else:
            raise ValueError('Invalid Decoder: ' + str(self.decoder))

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
        if self.decoder == 'refine':
            l1 = self.refine_conv1(l1)
            l2 = self.refine_conv2(l2)
            l3 = self.refine_conv3(l3)
            l4 = self.refine_conv4(l4)

            y4 = self.refine4(l4)          # [B, 512, h/32, w/32]
            y3 = self.refine3(y4, l3)   # [B, 256, h/16, w/16]
            y2 = self.refine2(y3, l2)   # [B, 256, h/8, w/8]
            y1 = self.refine1(y2, l1)   # [B, 256, h/4, w/4]
        elif self.decoder == 'base':
            y4 = self.up4(l4)          # [B, 256, h/16, w/16]
            y3 = self.refine4(y4, l3)  # [B, 256, h/16, w/16]

            y3 = self.up3(y3)          # [B, 128, h/8, w/8]
            y2 = self.refine3(y3, l2)  # [B, 128, h/8, w/8]

            y2 = self.up2(y2)          # [B, 64, h/4, w/4]
            y1 = self.refine2(y2, l1)  # [B, 64, h/4, w/4]
        else:
            raise ValueError('Invalid Decoder: ' + str(self.decoder))

        out = self.out_conv(y1)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out

def get_basenet(dataset='nyud', backbone='resnet18', pretrained=True, root='./encoding/models/pretrain/', module_setting={}):
    from ...datasets import datasets
    model = BaseNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, pretrained=pretrained, root=root, module_setting=module_setting)
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

class LearnedUpUnit(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.dep_conv = nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1, padding=1, groups=in_feats, bias=False)

    def forward(self, x):
        x = self.up(x)
        x = self.dep_conv(x)
        return x

class Simple_RGBD_Fuse(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        
    def forward(self, x, d):
        return x+d

class Simple_Level_Fuse(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        
    def forward(self, x, y):
        return x+y

class CBR(nn.Module):
    def __init__(self, in_feats, out_feats, k=2):
        super().__init__()
        self.mid_cbr = nn.Sequential(nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(in_feats), nn.ReLU(inplace=True)) if k != 1 else None
        self.out_cbr = nn.Sequential(nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_feats), nn.ReLU(inplace=True))
        
    def forward(self, x):
        if self.mid_cbr is not None:
            x = self.mid_cbr(x)
        return self.out_cbr(x)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, upsample=False):
        super(BasicBlock, self).__init__()
        self.upsample = upsample
        self.CBR1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(planes),
                                  nn.ReLU(inplace=True))

        if not upsample:
            self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(planes))
        else:
            self.conv2=nn.Sequential(nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=2, padding=0, bias=False),
                                     nn.BatchNorm2d(planes))
            self.up = nn.Sequential(nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=2, padding=0, bias=False),
                                    nn.BatchNorm2d(planes))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.upsample:
            identity = self.up(x)
        else:
            identity = x

        out = self.CBR1(x)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)

        return out

class FuseBlock(nn.Module):
    def __init__(self, in_chs, out_ch):
        # in_chs=[g_ch, x_ch]: g_ch 为深层特征的channel数， x_ch is num of channels for features from encoder
        super().__init__()
        assert len(in_chs)==2, 'provide num of channels for both inputs'
        g_ch, x_ch = in_chs
        self.g_rcu0 = BasicBlock(g_ch, g_ch)
        self.x_rcu0 = BasicBlock(x_ch, x_ch)

        self.g_rcu1 = BasicBlock(g_ch, out_ch, upsample=True)
        self.x_rcu1 = BasicBlock(x_ch, out_ch, upsample=False)

        self.g_conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.x_conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g = self.g_rcu1(self.g_rcu0(g))
        x = self.x_rcu1(self.x_rcu0(x))
        g = self.g_conv2(g)
        x = self.x_conv2(x)
        out = self.relu(g+x)
        return out

class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MRF_Concat_5_2, *shapes)