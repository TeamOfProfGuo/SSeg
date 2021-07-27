
import torch
import torch.nn as nn
from torch.nn import functional as F

from net.nn.gf import *
from net.utils.feat_op import *

__all__ = ['Decoder']

class Decoder(nn.Module):
    def __init__(self, decoder_feat, n_classes, decoder='base', decoder_args={}):
        super().__init__()
        decoder_dict = {'base': Base_Decoder, 'refine': Refine_Decoder, 'gf': GF_Decoder}
        self.decoder = decoder_dict[decoder](decoder_feat, n_classes, **decoder_args)
    
    def forward(self, feats):
        return self.decoder(feats)

class Base_Decoder(nn.Module):
    def __init__(self, decoder_feat, n_classes, conv_module='cbr', level_fuse='add', feats='f', rf_conv=(True, False), lf_bb='rbb', lf_args={}):
        super().__init__()

        self.feats = feats
        # level_fuse_dict = {'add': Simple_Level_Fuse, 'na': Norm_Add,'max': Max_Level_Fuse, 'gau': GAU_Block}

        # Refine Blocks
        for i in range(len(decoder_feat['level'])):
            self.add_module('refine%d' % i, Base_Level_Fuse(decoder_feat['level'][i], level_fuse, rf_conv, lf_bb, lf_args))

        # Upsample Blocks
        for i in range(len(decoder_feat['level'])):
            self.add_module('up%d' % i, up_block(decoder_feat['level'][i], conv_module))

        self.out_conv = out_block(min(decoder_feat['level']), decoder_feat['final'], n_classes, conv_module)

    def forward(self, feats):
        if self.feats == 'l':
            x1, x2, x3, x4 = feats.l1, feats.l2, feats.l3, feats.l4
        elif self.feats == 'd':
            x1, x2, x3, x4 = feats.d1, feats.d2, feats.d3, feats.d4
        elif self.feats == 'f':
            x1, x2, x3, x4 = feats.f1, feats.f2, feats.f3, feats.f4
        else:
            raise ValueError('Invalid out feats: %s.' % self.feats)

        feats = self.refine0(self.up0(x4), x3)
        feats = self.refine1(self.up1(feats), x2)
        feats = self.refine2(self.up2(feats), x1)
        return self.out_conv(feats)

class Refine_Decoder(nn.Module):
    def __init__(self, decoder_feat, n_classes, feats='f'):
        super().__init__()

        self.feats = feats
        n_features = decoder_feat['final']
        feats_list = decoder_feat['level']

        self.refine_conv1 = nn.Conv2d(feats_list[2], n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.refine_conv2 = nn.Conv2d(feats_list[1], n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.refine_conv3 = nn.Conv2d(feats_list[0], n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.refine_conv4 = nn.Conv2d(2*max(feats_list), 2*n_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.refine1 = RefineNetBlock(n_features, (n_features, 8), (n_features, 4))
        self.refine2 = RefineNetBlock(n_features, (n_features, 16), (n_features, 8))
        self.refine3 = RefineNetBlock(n_features, (2*n_features, 32), (n_features, 16))
        self.refine4 = RefineNetBlock(2*n_features, (2*n_features, 32))

        self.out_conv = nn.Sequential(
            ResidualConvUnit(n_features), ResidualConvUnit(n_features),
            nn.Conv2d(n_features, n_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, feats):
        if self.feats == 'l':
            x1, x2, x3, x4 = feats.l1, feats.l2, feats.l3, feats.l4
        elif self.feats == 'd':
            x1, x2, x3, x4 = feats.d1, feats.d2, feats.d3, feats.d4
        elif self.feats == 'f':
            x1, x2, x3, x4 = feats.f1, feats.f2, feats.f3, feats.f4
        else:
            raise ValueError('Invalid out feats: %s.' % self.feats)

        x1 = self.refine_conv1(x1)
        x2 = self.refine_conv2(x2)
        x3 = self.refine_conv3(x3)
        x4 = self.refine_conv4(x4)

        y4 = self.refine4(x4)       # [B, 512, h/32, w/32]
        y3 = self.refine3(y4, x3)   # [B, 256, h/16, w/16]
        y2 = self.refine2(y3, x2)   # [B, 256, h/8, w/8]
        y1 = self.refine1(y2, x1)   # [B, 256, h/4, w/4]

        return self.out_conv(y1)

class GF_Decoder(nn.Module):
    def __init__(self, n_features: int, n_classes: int, feats='f'):
        super().__init__()
        self.feats = feats
        ch_list = [(64, 128), (128, 256), (256, 512)]
        for i, (l, h) in enumerate(ch_list):
            self.add_module('refine%d' % (i+1), GF_Refiner(l, h))
        self.out_conv = nn.Sequential(
            CBR(64, n_features), CBR(n_features, n_features),
            nn.Conv2d(n_features, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, feats):
        if self.feats == 'l':
            x1, x2, x3, x4 = feats.l1, feats.l2, feats.l3, feats.l4
        elif self.feats == 'd':
            x1, x2, x3, x4 = feats.d1, feats.d2, feats.d3, feats.d4
        elif self.feats == 'f':
            x1, x2, x3, x4 = feats.f1, feats.f2, feats.f3, feats.f4
        else:
            raise ValueError('Invalid out feats: %s.' % self.feats)

        x3 = self.refine3(x3, x4)   # [B, 256, h/16, w/16]
        x2 = self.refine2(x2, x3)   # [B, 128, h/8, w/8]
        x1 = self.refine1(x1, x2)   # [B, 64, h/4, w/4]
        return self.out_conv(x1)

class GF_Refiner(nn.Module):
    def __init__(self, l: int, h: int):
        super().__init__()
        self.in_rcu1 = CBR(l, l) # RC_Block(l, unit_num=2)
        self.in_rcu2 = CBR(h, l) # RC_Block(h, unit_num=2)
        self.gf = GF_Module(l, h)
        self.out_rcu = RC_Block(l, unit_num=1)

    def forward(self, l, h):
        l = self.in_rcu1(l)
        h = self.in_rcu2(h)
        out = self.gf(l, h)
        return self.out_rcu(out[0])

class LearnedUpUnit(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.dep_conv = nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1, padding=1, groups=in_feats, bias=False)

    def forward(self, x):
        x = self.up(x)
        x = self.dep_conv(x)
        return x

class Simple_Level_Fuse(nn.Module):
    def __init__(self, in_feats, **kwargs):
        super().__init__()
        
    def forward(self, x, y):
        return x+y

class Norm_Add(nn.Module):
    def __init__(self, in_feats, **kwargs):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_feats)
        self.norm2 = nn.BatchNorm2d(in_feats)
        
    def forward(self, x, y):
        return self.norm1(x) + self.norm2(y)

class Max_Level_Fuse(nn.Module):
    def __init__(self, in_feats, **kwargs):
        super().__init__()
        
    def forward(self, x, y):
        return torch.max(x, y)

class CC1_Level_Fuse(nn.Module):
    def __init__(self, in_feats, **kwargs):
        super().__init__()
        self.cc_block = nn.Sequential(
            nn.BatchNorm2d(2 * in_feats), nn.ReLU(inplace=True),
            nn.Conv2d(2 * in_feats, in_feats, kernel_size=1, stride=1, padding=0, bias=False)
        )
        
    def forward(self, x, y):
        return self.cc_block(torch.cat((x, y), dim=1))

class CC2_Level_Fuse(nn.Module):
    def __init__(self, in_feats, **kwargs):
        super().__init__()
        self.cc_block = nn.Sequential(
            nn.Conv2d(2 * in_feats, in_feats, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_feats),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, y):
        return self.cc_block(torch.cat((x, y), dim=1))

class CC3_Level_Fuse(nn.Module):
    def __init__(self, in_feats, **kwargs):
        super().__init__()
        self.cc_block = nn.Sequential(
            nn.Conv2d(2 * in_feats, in_feats, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, y):
        return self.cc_block(torch.cat((x, y), dim=1))

class CC3I_Level_Fuse(nn.Module):
    def __init__(self, in_feats, pad_mode='zero', kernel=1, act='relu', init_args={}):
        super().__init__()
        pad_size = tuple([kernel // 2] * 4)
        act_dict = {'idt': nn.Identity(), 'relu': nn.ReLU(inplace=True)}

        pad_layer = nn.ZeroPad2d(pad_size) if pad_mode == 'zero' else nn.ReplicationPad2d(pad_size)
        conv_layer = nn.Conv2d(2*in_feats, in_feats, kernel_size=kernel, padding=0, groups=in_feats, bias=True)
        conv_layer.weight.data = init_conv(in_feats, 2, kernel, **init_args)
        act_layer = act_dict[act]

        self.rcci = nn.Sequential(pad_layer, conv_layer, act_layer)
        
    def forward(self, x, y):
        b, c, h, w = x.size()
        feats = torch.cat((x, y), dim=-2).reshape(b, 2*c, h, w)   # [b, c, 2h, w] => [b, 2c, h, w]
        return self.rcci(feats)

class Base_Level_Fuse(nn.Module):
    def __init__(self, in_feats, fuse_mode='na', conv_flag=(True, False), lf_bb='rbb', lf_args={}):
        super().__init__()
        self.conv_flag = conv_flag
        bb_dict = {'rbb': ResidualBasicBlock, 'drb': DepResidualBasicBlock}
        fuse_dict = {'add': Simple_Level_Fuse, 'na': Norm_Add, 'max': Max_Level_Fuse,
                     'cc1': CC1_Level_Fuse, 'cc2': CC2_Level_Fuse, 'cc3': CC3_Level_Fuse,
                     'cc3i': CC3I_Level_Fuse}
        self.fuse = fuse_dict[fuse_mode](in_feats, **lf_args)
        self.rbb0 = bb_dict[lf_bb](in_feats) if conv_flag[0] else nn.Identity()
        self.rbb1 = bb_dict[lf_bb](in_feats) if conv_flag[1] else nn.Identity()
    
    def forward(self, x, y):
        y = self.rbb0(y)    # Refine feats from backbone
        return self.rbb1(self.fuse(x, y))

class GAU_Block(nn.Module):
    def __init__(self, in_feats, r=16):
        super().__init__()
        # 参考PAN x 为浅层网络，y为深层网络
        self.x_conv = nn.Sequential(nn.Conv2d(in_feats, in_feats, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_feats))

        self.y_gap = nn.AdaptiveAvgPool2d(1)
        self.y_conv = nn.Sequential(nn.Conv2d(in_feats, in_feats, kernel_size=1, padding=0, bias=False),
                                    nn.BatchNorm2d(in_feats),
                                    nn.ReLU(inplace=True))

    def forward(self, y, x):
        x1 = self.x_conv(x)      # [B, c, h, w]

        y1 = self.y_gap(y)       # [B, c, 1, 1]
        y1 = self.y_conv(y1)     # [B, c, 1, 1]

        out = y1*x1 + y
        return out

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

class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x

class MRF_Concat_5_2(nn.Module):
    def __init__(self, out_feats, *shapes):
        super().__init__()
        _, min_scale = min(shapes, key=lambda x: x[1])

        self.scale_factors = []
        self.concat_count = 0
        for shape in shapes:
            self.concat_count += shape[0]
            self.scale_factors.append(shape[1] // min_scale)
        
        self.out_block = nn.Sequential(
            nn.BatchNorm2d(self.concat_count), nn.ReLU(inplace=True),
            nn.Conv2d(self.concat_count, out_feats, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, *xs):

        concat_feat = [xs[0]]
        if self.scale_factors[0] != 1:
            concat_feat[0] = nn.functional.interpolate(concat_feat[0], scale_factor=self.scale_factors[0], mode='bilinear', align_corners=True)

        for i, x in enumerate(xs[1:], 1): # the value for i starts from 1
            if self.scale_factors[i] != 1:
                x = nn.functional.interpolate(x, scale_factor=self.scale_factors[i], mode='bilinear', align_corners=True)
            concat_feat.append(x)
        
        output = torch.cat(tuple(concat_feat), 1)
        output = self.out_block(output)
        return output

class BaseRefineNetBlock(nn.Module):
    def __init__(self, features, residual_conv_unit, multi_resolution_fusion, *shapes):
        super().__init__()

        for i, shape in enumerate(shapes):
            feats = shape[0]
            self.add_module("rcu{}".format(i), nn.Sequential(residual_conv_unit(feats), residual_conv_unit(feats)))

        self.mrf = multi_resolution_fusion(features, *shapes) if len(shapes) != 1 else None
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        rcu_xs = []
        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__("rcu{}".format(i))(x))

        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]

        return self.output_conv(out)

class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MRF_Concat_5_2, *shapes)