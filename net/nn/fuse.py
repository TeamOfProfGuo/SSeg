
import torch
import torch.nn as nn
from torch.nn import functional as F

from .gf import GF_Module
from net.utils.feat_op import interpolate

__all__ = ['Fuse_Block', 'Simple_RGBD_Fuse', 'RGBD_Fuse_Block', 'GAU_Fuse', 
           'PPE_Block', 'PPCE_Block', 'PDLC_Fuse']

PCA_FUSE_LAMB = False

class Fuse_Block(nn.Module):
    def __init__(self, in_feats, fb='simple', fuse_args={}):
        super().__init__()
        fuse_dict = {'simple': Simple_RGBD_Fuse, 'fuse': RGBD_Fuse_Block, 'gau': GAU_Fuse, 
                     'gf': GF_Module, 'pdlc': PDLC_Fuse, 'lgc': LGC_Fuse, 'cc': CC_Fuse,
                     'rcc': RCC_Fuse}
        self.fb = fuse_dict[fb](in_feats, **fuse_args)
        
    def forward(self, rgb, dep):
        return self.fb(rgb, dep)

class Simple_RGBD_Fuse(nn.Module):
    def __init__(self, in_feats, **kwargs):
        super().__init__()
        
    def forward(self, x, d):
        return x+d, d

class CC3_Block(nn.Sequential):
    def __init__(self, in_feats):
        super().__init__()
        self.add_module('cc', nn.Sequential(
            nn.Conv2d(2 * in_feats, in_feats, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        ))

class RCC_Block(nn.Module):
    def __init__(self, in_feats, kernel=1, act='idt'):
        super().__init__()
        act_dict = {'idt': nn.Identity(), 'relu': nn.ReLU(inplace=True)}
        self.rcc = nn.Sequential(
            nn.Conv2d(2*in_feats, in_feats, kernel_size=kernel, padding=kernel//2, groups=in_feats, bias=True),
            act_dict[act]
        )
        
    def forward(self, x, d):
        b, c, h, w = x.size()
        y = torch.cat((x, d), dim=-2).reshape(b, 2*c, h, w)   # [b, c, 2h, w] => [b, 2c, h, w]
        return self.rcc(y)

class RGBD_Fuse_Block(nn.Module):
    def __init__(self, in_feats, out_method='add', pre_module='pca', mode='late', pre_setting={},
                    use_lamb=True, refine_dep=False):
        super().__init__()
        module_dict = ATT_MODULE_DICT
        self.mode = mode
        self.module = module_dict[pre_module]
        self.out_method = out_method
        self.use_lamb = use_lamb
        self.refine_dep = refine_dep
        self.lamb = nn.Parameter(torch.zeros(1))
        # self.dep_bn = nn.BatchNorm2d(in_feats)
        self.gamma = nn.Parameter(torch.zeros(1))
        print('[RGB-D Fuse]: use_lamb =', self.use_lamb)
        if mode == 'late':
            self.rgb_pre = self.module(in_feats, **pre_setting)
            self.dep_pre = self.module(in_feats, **pre_setting)
        elif mode == 'early':
            self.pre = self.module(2 * in_feats, **pre_setting)
        else:
            raise ValueError('Invalid Fuse Mode: ' + str(mode))
        if out_method == 'concat':
            self.out_block = nn.Sequential(
                nn.Conv2d(2 * in_feats, in_feats, kernel_size=1, stride=1, bias=False)
            )

    def forward(self, rgb, dep):
        d = dep.clone()
        if self.mode == 'late':
            rgb = self.rgb_pre(rgb)
            dep = self.dep_pre(dep)
            if self.out_method == 'add':
                out = rgb + self.lamb * dep if self.use_lamb else rgb + dep
            elif self.out_method == 'max':
                out = torch.max(rgb, dep)
            else:
                out = self.out_block(torch.cat((rgb, dep), 1))
        elif self.mode == 'early':
            _, c, _, _ = rgb.size()
            feats = torch.cat((rgb, dep), dim=1)
            feats = self.pre(feats)
            if self.out_method == 'add':
                rgb = feats[:, :c, :, :]
                dep = feats[:, c:, :, :]
                out = rgb + self.lamb * dep if self.use_lamb else rgb + dep
            else:
                out =  self.out_block(feats)
        else:
            raise ValueError('Invalid Fuse Mode: ' + str(self.mode))

        return out, (dep if self.refine_dep else d)

class CC_Fuse(nn.Module):
    def __init__(self, in_feats, mode='rsp', att_module='pdl', att_setting={}, refine_dep=False):
        super().__init__()
        module_dict = ATT_MODULE_DICT
        self.mode = mode
        self.refine_dep = refine_dep
        self.cc1 = CC3_Block(in_feats)
        self.cc2 = CC3_Block(in_feats) if refine_dep else None
        if mode == 'rsp':
            self.rgb_pre = module_dict[att_module](in_feats, **att_setting)
            self.dep_pre = module_dict[att_module](in_feats, **att_setting)
        else:
            self.pre = module_dict[att_module](2 * in_feats, **att_setting)
    
    def forward(self, rgb, dep):
        d = dep.clone()
        if self.mode == 'rsp':
            rgb = self.rgb_pre(rgb)
            dep = self.dep_pre(dep)
            feats = torch.cat((rgb, dep), dim=1)
        else:
            feats = torch.cat((rgb, dep), dim=1)
            feats = self.pre(feats)
        return self.cc1(feats), (self.cc2(feats) if self.refine_dep else d)

class RCC_Fuse(nn.Module):
    def __init__(self, in_feats, fuse_setting={}, att_module='pdl', att_setting={}, refine_dep=False):
        super().__init__()
        module_dict = ATT_MODULE_DICT
        self.refine_dep = refine_dep
        self.rcc1 = RCC_Block(in_feats, **fuse_setting)
        self.rcc2 = RCC_Block(in_feats, **fuse_setting) if refine_dep else None
        self.rgb_pre = module_dict[att_module](in_feats, **att_setting)
        self.dep_pre = module_dict[att_module](in_feats, **att_setting)
    
    def forward(self, rgb, dep):
        d = dep.clone()
        rgb = self.rgb_pre(rgb)
        dep = self.dep_pre(dep)
        rgb_out = self.rcc1(rgb, dep)
        dep_out = self.rcc2(rgb, dep) if self.refine_dep else d
        return rgb_out, dep_out

class PDLC_Fuse(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.feats = in_feats
        self.rgb_norm = nn.BatchNorm2d(in_feats)
        self.dep_norm = nn.BatchNorm2d(in_feats)
        self.att = PDLW_Block(2 * in_feats)
        self.lamb = nn.Parameter(torch.zeros(1))
        self.dep_conv = nn.Conv2d(in_feats, in_feats, kernel_size=1, bias=False)

        # self.rgb_att = PDLW_Block(in_feats)
        # self.dep_att = PDLW_Block(in_feats)
        # self.rgb_conv = nn.Sequential(
        #     nn.Conv2d(in_feats, in_feats, kernel_size=1, padding=0, bias=False),
        #     nn.BatchNorm2d(in_feats),
        #     nn.ReLU(inplace=True)
        # )
        # self.dep_conv = nn.Sequential(
        #     nn.Conv2d(in_feats, in_feats, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(in_feats)
        # )

    def forward(self, rgb, dep):
        rgb = self.rgb_norm(rgb)
        dep = self.dep_norm(dep)
        w = self.att(torch.cat((rgb, dep), dim=1))
        feats = w * torch.cat((rgb, dep), dim=1)
        dep_out = feats[:, self.feats:, :, :]
        rgb_out = feats[:, :self.feats, :, :] + self.lamb * self.dep_conv(dep_out)
        return rgb_out, dep_out
        # rgb_w = self.rgb_att(rgb)
        # dep_w = self.dep_att(dep)
        # dep_out = dep_w * dep
        # rgb_out = rgb_w * rgb + self.rgb_conv(rgb_w) * self.dep_conv(dep_out)
        # return rgb_out, dep_out

class Identity_ARM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x):
        return x

class PDLW_Block(nn.Module):
    def __init__(self, in_feats, pp_layer=4, descriptor=8, mid_feats=16, act_layer=nn.Sigmoid()):
        super().__init__()
        self.layer_size = pp_layer                  # l: pyramid layer num
        self.feats_size = (4 ** pp_layer - 1) // 3  # f: feats for descritor
        self.descriptor = descriptor                # d: descriptor num (for one channel)
        print('[PDLW]: l = %d, d = %d, m = %d.' % (pp_layer, descriptor, mid_feats))

        self.des = nn.Conv2d(self.feats_size, descriptor, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(descriptor * in_feats, mid_feats, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_feats, in_feats),
            act_layer
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        l, f, d = self.layer_size, self.feats_size, self.descriptor
        pooling_pyramid = []
        for i in range(l):
            pooling_pyramid.append(F.adaptive_avg_pool2d(x, 2 ** i).view(b, c, 1, -1))
        y = torch.cat(tuple(pooling_pyramid), dim=-1)   # [b,  c, 1, f]
        y = y.reshape(b*c, f, 1, 1)                     # [bc, f, 1, 1]
        y = self.des(y).view(b, c*d)                    # [bc, d, 1, 1] => [b, cd, 1, 1]
        w = self.mlp(y).view(b, c, 1, 1)                # [b,  c, 1, 1] => [b, c, 1, 1]
        return w

class PDLD_Block(nn.Module):
    def __init__(self, in_feats, pp_layer=4, descriptor=8):
        super().__init__()
        self.layer_size = pp_layer                  # l: pyramid layer num
        self.feats_size = (4 ** pp_layer - 1) // 3  # f: feats for descritor
        self.descriptor = descriptor                # d: descriptor num (for one channel)
        print('[PDLD]: l = %d, d = %d.' % (pp_layer, descriptor))

        self.des = nn.Conv2d(self.feats_size, descriptor, kernel_size=1)
        
    def forward(self, x):
        b, c, _, _ = x.size()
        l, f, d = self.layer_size, self.feats_size, self.descriptor
        pooling_pyramid = []
        for i in range(l):
            pooling_pyramid.append(F.adaptive_avg_pool2d(x, 2 ** i).view(b, c, 1, -1))
        y = torch.cat(tuple(pooling_pyramid), dim=-1)   # [b,  c, 1, f]
        y = y.reshape(b*c, f, 1, 1)                     # [bc, f, 1, 1]
        des = self.des(y).view(b, c*d, 1, 1)            # [bc, d, 1, 1] => [b, cd, 1, 1]
        return des

class GAU_Fuse(nn.Module):
    def __init__(self, in_feats, refine_rgb=True, refine_dep=False, gau_args={}):
        super().__init__()
        self.rgb_ref = GAU_Block(in_feats, **gau_args) if refine_rgb else Identity_Block()
        self.dep_ref = GAU_Block(in_feats, **gau_args) if refine_dep else Identity_Block()

    def forward(self, rgb, dep):
        return self.rgb_ref(rgb, dep), self.dep_ref(dep, rgb)

class LGC_Fuse(nn.Module):
    def __init__(self, in_feats, refine_rgb=True, refine_dep=False):
        super().__init__()
        self.refine_rgb = refine_rgb
        self.refine_dep = refine_dep
        self.lamb1 = nn.Parameter(torch.zeros(1))
        self.lamb2 = nn.Parameter(torch.zeros(1))
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(in_feats, 1, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.dep_conv = nn.Sequential(
            nn.Conv2d(in_feats, 1, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, rgb, dep):
        rgb_out, dep_out = rgb.clone(), dep.clone()
        if self.refine_rgb:
            dep_context = self.dep_conv(dep)
            rgb_out = rgb + self.lamb1 * dep_context
        if self.refine_dep:
            rgb_context = self.rgb_conv(rgb)
            dep_out = dep + self.lamb2 * rgb_context
        return rgb_out, dep_out

class Identity_Block(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y, x):
         return y

class GAU_Block(nn.Module):
    def __init__(self, in_feats, use_lamb=False):
        super().__init__()
        
        self.lamb = nn.Parameter(torch.zeros(1)) if use_lamb else 1
        self.use_lamb = use_lamb
    
        # 参考PAN x 为浅层网络，y为深层网络 => x(dep), y(rgb)
        self.x_conv = nn.Sequential(nn.Conv2d(in_feats, in_feats, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_feats))
        self.y_gap = nn.AdaptiveAvgPool2d(1)
        self.y_conv = nn.Sequential(nn.Conv2d(in_feats, in_feats, kernel_size=1, padding=0, bias=False),
                                    nn.BatchNorm2d(in_feats),
                                    nn.ReLU(inplace=True))

    def forward(self, y, x):
        x = self.x_conv(x)                  # [B, c, h, w]
        w = self.y_conv(self.y_gap(y))      # [B, c, 1, 1]
        out = y + self.lamb * (w * x)
        return out

class PP_Block(nn.Module):
    def __init__(self, in_feats, pp_layer=3, reduction=16, mid_feats=32):
        super().__init__()
        self.pp_layer = pp_layer            # l: pyramid layer num
        self.pp_size = 2 ** (pp_layer-1)    # s: pyramid max-layer size
        self.pp_des = self.pp_size ** 2     # d: pyramid max-layer feature num
        self.pp_conv = nn.Conv2d(in_feats * pp_layer, in_feats, kernel_size=1, groups=in_feats)
        self.fc_feats = self.pp_des * in_feats
        self.fc = nn.Sequential(
            # PP Var1
            nn.Linear(self.fc_feats, in_feats, bias=False),

            # # PP Var2
            # nn.ReLU(inplace=True),
            # nn.Linear(self.fc_feats, in_feats, bias=False),

            # # PP Var3
            # nn.Linear(self.fc_feats, self.fc_feats // reduction, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Linear(self.fc_feats // reduction, in_feats, bias=False),

            # # PP Var4
            # nn.Linear(self.fc_feats, mid_feats, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Linear(mid_feats, in_feats, bias=False),
            
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        l, s = self.pp_layer, self.pp_size
        pooling_pyramid = []
        for i in range(self.pp_layer):
            pooling_pyramid.append(F.interpolate(F.adaptive_avg_pool2d(x, 2 ** i), size=self.pp_size))
        y = torch.cat(tuple(pooling_pyramid), dim=-1)   # [b, c, s, sl]
        y = y.permute(0, 1, 3, 2).reshape(b, c*l, s, s) # [b, c, sl, s] => [b, cl, s, s]
        y = self.pp_conv(y).view(b, -1)                 # [b, c, s, s] => [b, cs^2]
        weighting = self.fc(y).view(b, c, 1, 1)         # [b, c] => [b, c, 1, 1]
        return weighting * x

class PPC_Block(nn.Module):
    def __init__(self, in_feats, pp_layer=3, mid_feats=32, reduction=8):
        super().__init__()
        self.pp_layer = pp_layer            # l: pyramid layer num
        self.pp_size = 2 ** (pp_layer-1)    # s: pyramid max-layer size

        # # Var 1
        # self.cap_conv = nn.Conv2d(in_feats * pp_layer, in_feats, kernel_size=1)

        # Var 2
        self.mlp = nn.Sequential(
            nn.Conv2d(in_feats * pp_layer, in_feats // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_feats // reduction, in_feats, kernel_size=1)
        )

        # # Var 3
        # self.cap_conv = nn.Conv2d(in_feats * pp_layer, 1, kernel_size=1)

        self.att_conv = nn.Conv2d(1, 1, kernel_size=self.pp_size)
        
    def forward(self, x):
        b, c, _, _ = x.size()
        l, s = self.pp_layer, self.pp_size
        pooling_pyramid = []
        for i in range(self.pp_layer):
            pooling_pyramid.append(F.interpolate(F.adaptive_avg_pool2d(x, 2 ** i), size=self.pp_size))
        y = torch.cat(tuple(pooling_pyramid), dim=-1)          # [b, c, s, sl]
        y = y.permute(0, 1, 3, 2).reshape(b, c*l, s, s)        # [b, c, sl, s] => [b, cl, s, s]
        # y = y.permute(0, 2, 3, 1).reshape(b*s*s, c*l, 1, 1)  # [b, s, s, cl] => [bs^2, cl, 1, 1]
        y = self.mlp(y).view(b*c, 1, s, s)                     # [b, c, s, s]  => [bc, 1, s, s]
        w = torch.sigmoid(self.att_conv(y).view(b, c, 1, 1))   # [bc, 1, 1, 1] => [b, c, 1, 1]
        return w * x

class PDL_Block(nn.Module):
    def __init__(self, in_feats, pp_layer=4, descriptor=8, reduction=16, mid_feats=16):
        super().__init__()
        self.layer_size = pp_layer                  # l: pyramid layer num
        self.feats_size = (4 ** pp_layer - 1) // 3  # f: feats for descritor
        self.descriptor = descriptor                # d: descriptor num (for one channel)
        # print('[PDL]: l = %d, d = %d, r = %d.' % (pp_layer, descriptor, reduction))
        print('[PDL]: l = %d, d = %d, m = %d.' % (pp_layer, descriptor, mid_feats))

        self.des = nn.Conv2d(self.feats_size, descriptor, kernel_size=1)
        # Var 1
        self.mlp = nn.Sequential(
            # descriptor * in_feats // reduction
            # nn.Dropout(0.1, inplace=True),
            nn.Linear(descriptor * in_feats, mid_feats, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_feats, in_feats),
            nn.Sigmoid()
        )

        # # Var 2
        # self.mlp = nn.Sequential(
        #     nn.Linear(descriptor * in_feats, in_feats, bias=False),
        #     nn.Sigmoid()
        # )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        l, f, d = self.layer_size, self.feats_size, self.descriptor
        pooling_pyramid = []
        for i in range(l):
            pooling_pyramid.append(F.adaptive_avg_pool2d(x, 2 ** i).view(b, c, 1, -1))
        y = torch.cat(tuple(pooling_pyramid), dim=-1)   # [b,  c, 1, f]
        y = y.reshape(b*c, f, 1, 1)                     # [bc, f, 1, 1]
        y = self.des(y).view(b, c*d)                    # [bc, d, 1, 1] => [b, cd, 1, 1]
        w = self.mlp(y).view(b, c, 1, 1)                # [b,  c, 1, 1] => [b, c, 1, 1]
        return w * x

class PPE_Block(nn.Module):
    def __init__(self, in_feats, pp_layer=3, mid_feats=16):
        super().__init__()
        self.pp_layer = pp_layer            # l: pyramid layer num
        self.pp_size = 2 ** (pp_layer-1)    # s: pyramid max-layer size
        self.pp_des = self.pp_size ** 2     # d: pyramid max-layer feature num

        self.rgb_pp_conv = nn.Conv2d(in_feats * pp_layer, in_feats, kernel_size=1, groups=in_feats)
        self.dep_pp_conv = nn.Conv2d(in_feats * pp_layer, in_feats, kernel_size=1, groups=in_feats)
        self.fc_feats = 2 * self.pp_des * in_feats
        self.fc = nn.Sequential(
            # nn.Dropout(0.1, inplace=True),
            # # PP Var1
            # nn.Linear(self.fc_feats, 2 * in_feats, bias=False),
            # # PP Var2
            # nn.ReLU(inplace=True),
            # nn.Linear(self.fc_feats, 2 * in_feats, bias=False),
            # PP Var3
            nn.Linear(self.fc_feats, mid_feats, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_feats, 2 * in_feats),
            nn.Sigmoid()
        )

        self.pam = PSC_Block(in_feats)

    def forward(self, rgb, dep):
        b, c, _, _ = rgb.size()
        l, s = self.pp_layer, self.pp_size
        rgb_pp, dep_pp = [], []
        for i in range(self.pp_layer):
            rgb_pp.append(F.interpolate(F.adaptive_avg_pool2d(rgb, 2 ** i), size=self.pp_size))
            dep_pp.append(F.interpolate(F.adaptive_avg_pool2d(dep, 2 ** i), size=self.pp_size))
        rgb_y = torch.cat(tuple(rgb_pp), dim=-1)                                  # [b, c, s, sl]
        dep_y = torch.cat(tuple(dep_pp), dim=-1)                                  # [b, c, s, sl]
        rgb_y = rgb_y.permute(0, 1, 3, 2).reshape(b, c*l, s, s)                   # [b, c, sl, s] => [b, cl, s, s]
        dep_y = dep_y.permute(0, 1, 3, 2).reshape(b, c*l, s, s)                   # [b, c, sl, s] => [b, cl, s, s]
        rgb_y = self.rgb_pp_conv(rgb_y).view(b, -1)                               # [b, c, s, s] => [b, cs^2]
        dep_y = self.rgb_pp_conv(dep_y).view(b, -1)                               # [b, c, s, s] => [b, cs^2]
        weighting = self.fc(torch.cat((rgb_y, dep_y), dim=1)).view(b, 2*c, 1, 1)  # [b, 2c] => [b, 2c, 1, 1]
        out = weighting * torch.cat((rgb, dep), dim=1)
        # return out[:, :c, :, :] + out[:, c:, :, :]
        y = out[:, :c, :, :] + out[:, c:, :, :]
        return self.pam(y)

class PPCE_Block(nn.Module):
    def __init__(self, in_feats, pp_layer=3, mid_feats=32, reduction=16):
        super().__init__()
        self.pp_layer = pp_layer            # l: pyramid layer num
        self.pp_size = 2 ** (pp_layer-1)    # s: pyramid max-layer size

        # # Var 1
        # self.cap_conv = nn.Conv2d(2 * in_feats * pp_layer, 2 * in_feats, kernel_size=1)

        # Var 2
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * in_feats * pp_layer, in_feats // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_feats // reduction, 2 * in_feats, kernel_size=1)
        )

        self.att_conv = nn.Conv2d(1, 1, kernel_size=self.pp_size)
        
    def forward(self, rgb, dep):
        b, c, _, _ = rgb.size()
        l, s = self.pp_layer, self.pp_size
        rgb_pp, dep_pp = [], []
        for i in range(self.pp_layer):
            rgb_pp.append(F.interpolate(F.adaptive_avg_pool2d(rgb, 2 ** i), size=self.pp_size))
            dep_pp.append(F.interpolate(F.adaptive_avg_pool2d(dep, 2 ** i), size=self.pp_size))
        rgb_y = torch.cat(tuple(rgb_pp), dim=-1)            # [b, c, s, sl]
        dep_y = torch.cat(tuple(dep_pp), dim=-1)            # [b, c, s, sl]
        y = torch.cat((rgb_y, dep_y), dim=-1)               # [b, c, s, 2sl]
        y = y.permute(0, 1, 3, 2).reshape(b, 2*c*l, s, s)   # [b, c, 2sl, s] => [b, 2cl, s, s]
        # # Var 1
        # y = self.cap_conv(y).view(2*b*c, 1, s, s)                  # [b, 2c, s, s]  => [2bc, 1, s, s]
        # Var 2
        y = self.mlp(y).view(2*b*c, 1, s, s)                       # [b, 2c, s, s]  => [2bc, 1, s, s]
        w = torch.sigmoid(self.att_conv(y).view(b, 2*c, 1, 1))     # [2bc, 1, 1, 1] => [b, 2c, 1, 1]
        out = w * torch.cat((rgb, dep), dim=1)
        return out[:, :c, :, :] + out[:, c:, :, :]

class SC_Block(nn.Module):
    def __init__(self, in_feats, layers=(2, 4, 7)):
        super().__init__()
        self.layers = layers
        # Original
        for i in layers:
            layer = nn.Sequential(
                nn.Conv2d(in_feats, 1, kernel_size=i, stride=i//2, bias=False),
                nn.ReLU()
            )
            self.add_module('ss%d' % i, layer)
        
        self.att_conv = nn.Sequential(
            nn.Conv2d(len(layers), 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, _, h, w = x.size()
        sp_layers = []
        for i in self.layers:
            # l = F.interpolate(self.__getattr__('ss%d' % i)(x), size=(h, w))
            l = F.interpolate(self.__getattr__('ss%d' % i)(x), size=(h, w), mode='bilinear', align_corners=False)
            sp_layers.append(l)
        y = torch.cat(tuple(sp_layers), dim=1)
        w = self.att_conv(y)
        return w * x

class PSC_Block(nn.Module):
    def __init__(self, in_feats, pp_size=(1, 2, 4, 8), mid_feats=16):
        super().__init__()

        self.pp_size = pp_size
        self.ave_pool = True
        self.max_pool = False
        
        # # Var 1
        # lin_feats = sum([i**2 for i in pp_size])
        # out_feats = max(pp_size) ** 2
        # self.mlp = nn.Sequential(
        #     nn.Linear(lin_feats, mid_feats, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(mid_feats, out_feats, bias=False),
        #     nn.Sigmoid()
        # )

        # Var 2
        self.att_conv = nn.Sequential(
            nn.Conv2d(len(pp_size) * (self.ave_pool+self.max_pool), 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        sp_layers = []

        # # Var 1
        # for i in self.pp_size:
        #     l = F.adaptive_avg_pool2d(x, i).view(b, c, 1, -1)
        #     l = torch.mean(l, dim=1).view(b, -1)
        #     sp_layers.append(l)
        # y = torch.cat(tuple(sp_layers), dim=1)
        # paw = self.mlp(y).view(b, 1, max(self.pp_size), -1)

        # Var 2
        for i in self.pp_size:
            if self.ave_pool:
                l = torch.mean(F.adaptive_avg_pool2d(x, i), dim=1).view(b, 1, i, i)
                sp_layers.append(F.interpolate(l, size=max(self.pp_size)))
            if self.max_pool:
                l = torch.mean(F.adaptive_max_pool2d(x, i), dim=1).view(b, 1, i, i)
                sp_layers.append(F.interpolate(l, size=max(self.pp_size)))
        y = torch.cat(tuple(sp_layers), dim=1)
        paw = F.interpolate(self.att_conv(y), size=(h, w), mode='bilinear', align_corners=False)
        
        return paw * x

class PSCW_Block(nn.Module):
    def __init__(self, in_feats, pp_size=(1, 2, 4, 8), act_layer=nn.Sigmoid()):
        super().__init__()
        self.pp_size = pp_size
        self.att_conv = nn.Sequential(
            nn.Conv2d(len(pp_size), 1, kernel_size=1),
            act_layer
        )

    def forward(self, x):
        b, _, h, w = x.size()
        sp_layers = []
        for i in self.pp_size:
            l = torch.mean(F.adaptive_avg_pool2d(x, i), dim=1).view(b, 1, i, i)
            sp_layers.append(F.interpolate(l, size=max(self.pp_size)))
        y = torch.cat(tuple(sp_layers), dim=1)
        paw = F.interpolate(self.att_conv(y), size=(h, w), mode='bilinear', align_corners=False)
        return paw

class PCA_Block(nn.Module):
    def __init__(self, in_feats, mode='pc'):
        super().__init__()
        self.mode = mode
        self.pam = PSC_Block(in_feats)
        self.cam = PDL_Block(in_feats)
        if mode == 'add':
            self.use_lamb = PCA_FUSE_LAMB
            self.lamb = nn.Parameter(torch.zeros(1))
            print('[PCA]: use_lamb =', self.use_lamb)
    
    def forward(self, x):
        if self.mode == 'add':
            c = self.cam(x)
            p = self.pam(x)
            y = self.lamb * c + (1 - self.lamb) * p if self.use_lamb else c + p
        elif self.mode == 'max':
            c = self.cam(x)
            p = self.pam(x)
            y = torch.max(c, p)
        elif self.mode == 'pc':
            y = self.pam(x)
            y = self.cam(y)
        elif self.mode == 'cp':
            y = self.cam(x)
            y = self.pam(y)
        else:
            raise ValueError('Invalid PCA mode %s.' % self.mode)
        return y

class LGC_Block(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.lamb = nn.Parameter(torch.zeros(1))
        self.att_conv = nn.Sequential(
            nn.Conv2d(in_feats, 1, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
            # nn.Sigmoid()
        )
    
    def forward(self, x):
        return x + self.lamb * self.att_conv(x)

class PGC_Block(nn.Module):
    def __init__(self, in_feats, pp_size=(1, 3, 5, 7), act='sigmoid', conv='own', out='add'):
        super().__init__()
        self.out = out
        self.conv = conv
        self.pp_size = pp_size
        self.lamb = nn.Parameter(torch.zeros(1))
        self.gc_conv = nn.Conv2d(in_feats, 1, kernel_size=1, bias=False)
        for s in pp_size:
            if conv == 'own':
                self.add_module('pc%d' % s, nn.Sequential(
                    nn.AdaptiveAvgPool2d(s),
                    nn.Conv2d(in_feats, 1, kernel_size=1, bias=False)
                ))
            else:
                self.add_module('pool%d' % s, nn.AdaptiveAvgPool2d(s))
        self.att_conv = nn.Sequential(
            nn.Conv2d(len(pp_size)+1, 1, kernel_size=1),
            nn.Sigmoid() if act == 'sigmoid' else nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        _, _, h, w = x.size()
        gc_feats = [self.gc_conv(x)]
        for s in self.pp_size:
            if self.conv == 'own':
                layer = self.__getattr__('pc%d' % s)(x)
            else:
                layer = self.gc_conv(self.__getattr__('pool%d' % s)(x))
            gc_feats.append(interpolate(layer, (h, w), 'nearest'))
        pgc_map = self.att_conv(torch.cat(tuple(gc_feats), dim=1))
        return (x + self.lamb * pgc_map) if self.out == 'add' else (pgc_map * x)

class PGCW_Block(nn.Module):
    def __init__(self, in_feats, pp_size=(1, 3, 5, 7), act_layer=nn.Sigmoid()):
        super().__init__()
        self.pp_size = pp_size
        self.gc_conv = nn.Conv2d(in_feats, 1, kernel_size=1, bias=False)
        for s in pp_size:
            self.add_module('pc%d' % s, nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_feats, 1, kernel_size=1, bias=False)
            ))
        self.att_conv = nn.Sequential(
            nn.Conv2d(len(pp_size)+1, 1, kernel_size=1),
            act_layer
        )
    
    def forward(self, x):
        _, _, h, w = x.size()
        gc_feats = [self.gc_conv(x)]
        for s in self.pp_size:
            layer = self.__getattr__('pc%d' % s)(x)
            gc_feats.append(interpolate(layer, (h, w), 'nearest'))
        pgc_map = self.att_conv(torch.cat(tuple(gc_feats), dim=1))
        return pgc_map

class GAM_Block(nn.Module):
    def __init__(self, in_feats, cam_act='sigmoid', pam_act='relu'):
        super().__init__()
        act1 = nn.ReLU(inplace=True) if cam_act == 'relu' else nn.Sigmoid()
        act2 = nn.ReLU(inplace=True) if pam_act == 'relu' else nn.Sigmoid()   
        self.lamb = nn.Parameter(torch.zeros(1))
        self.cam = PDLW_Block(in_feats, act_layer=act1)
        self.pam = nn.Sequential(
            nn.Conv2d(in_feats, 1, kernel_size=1, bias=False),
            act2
        )
    
    def forward(self, x):
        gam = self.cam(x) * self.pam(x)
        return x + self.lamb * gam

class AGAM_Block(nn.Module):
    def __init__(self, in_feats, use_lamb=True, lamb=1):
        super().__init__()
        self.lamb = lamb
        self.cam = PDLW_Block(in_feats)
        self.pam = PSCW_Block(in_feats)
        if lamb == 1:
            self.lamb0 = nn.Parameter(torch.ones(1) / 2) if use_lamb else 0.5
        else:
            self.lamb1 = nn.Parameter(torch.ones(1)) if use_lamb else 1
            self.lamb2 = nn.Parameter(torch.ones(1)) if use_lamb else 1
    
    def forward(self, x):
        caw = self.cam(x)
        paw = self.pam(x)
        if self.lamb == 1:
            gaw = self.lamb0 * caw + (1 - self.lamb0) * paw
        else:
            gaw = self.lamb1 * caw + self.lamb2 * paw
        return gaw * x

class BGAM_Block(nn.Module):
    def __init__(self, in_feats, use_lamb=True, lamb=1):
        super().__init__()
        self.lamb = lamb
        self.cam = PDLW_Block(in_feats, act_layer=nn.Identity())
        self.pam = PSCW_Block(in_feats, act_layer=nn.Identity())
        if lamb == 1:
            self.lamb0 = nn.Parameter(torch.ones(1) / 2) if use_lamb else 0.5
        else:
            self.lamb1 = nn.Parameter(torch.ones(1)) if use_lamb else 1
            self.lamb2 = nn.Parameter(torch.ones(1)) if use_lamb else 1
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        caw = self.cam(x)
        paw = self.pam(x)
        if self.lamb == 1:
            gaw = self.lamb0 * caw + (1 - self.lamb0) * paw
        else:
            gaw = self.lamb1 * caw + self.lamb2 * paw
        return self.sigmoid(gaw) * x

class CGAM_Block(nn.Module):
    def __init__(self, in_feats, use_lamb=True, lamb=1):
        super().__init__()
        self.lamb = lamb
        self.cam = PDLW_Block(in_feats)
        self.pam = PGCW_Block(in_feats)
        if lamb == 1:
            self.lamb0 = nn.Parameter(torch.ones(1) / 2) if use_lamb else 0.5
        else:
            self.lamb1 = nn.Parameter(torch.ones(1)) if use_lamb else 1
            self.lamb2 = nn.Parameter(torch.ones(1)) if use_lamb else 1
    
    def forward(self, x):
        caw = self.cam(x)
        paw = self.pam(x)
        if self.lamb == 1:
            gaw = self.lamb0 * caw + (1 - self.lamb0) * paw
        else:
            gaw = self.lamb1 * caw + self.lamb2 * paw
        return gaw * x

class DGAM_Block(nn.Module):
    def __init__(self, in_feats, use_lamb=True, lamb=1):
        super().__init__()
        self.lamb = lamb
        self.cam = PDLW_Block(in_feats, act_layer=nn.Identity())
        self.pam = PGCW_Block(in_feats, act_layer=nn.Identity())
        if lamb == 1:
            self.lamb0 = nn.Parameter(torch.ones(1) / 2) if use_lamb else 0.5
        else:
            self.lamb1 = nn.Parameter(torch.ones(1)) if use_lamb else 1
            self.lamb2 = nn.Parameter(torch.ones(1)) if use_lamb else 1
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        caw = self.cam(x)
        paw = self.pam(x)
        if self.lamb == 1:
            gaw = self.lamb0 * caw + (1 - self.lamb0) * paw
        else:
            gaw = self.lamb1 * caw + self.lamb2 * paw
        return self.sigmoid(gaw) * x

class SAM_Block(torch.nn.Module):
    def __init__(self, in_feats, lamb=1e-4):
        super().__init__()
        self.act = nn.Sigmoid()
        self.lamb = lamb

    def forward(self, x):
        _, _, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.lamb)) + 0.5
        return x * self.act(y)

class SE_Block(nn.Module):
    def __init__(self, in_feats, mid_factor=16):
        super().__init__()
        self.mid_feats = in_feats // mid_factor
        self.fc = nn.Sequential(
            nn.Conv2d(in_feats, self.mid_feats, kernel_size=1),
            # nn.BatchNorm2d(self.mid_feats),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_feats, in_feats, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = weighting * x
        return y

class SPA_Block(nn.Module):
    def __init__(self, in_feats, reduction=16):
        super().__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        pool_feats_num = 21 * in_feats
        self.fc = nn.Sequential(
            nn.Linear(pool_feats_num, pool_feats_num // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(pool_feats_num // reduction, in_feats, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x).view(b, c)  # like resize() in numpy
        y2 = self.avg_pool2(x).view(b, 4 * c)
        y3 = self.avg_pool4(x).view(b, 16 * c)
        y = torch.cat((y1, y2, y3), 1)
        weighting = self.fc(y)
        b, out_channel = weighting.size()
        weighting = weighting.view(b, out_channel, 1, 1)
        # return weighting
        out = weighting * x
        return out

class GC_Block(nn.Module):
    def __init__(self, in_feats, mid_factor=4):
        super().__init__()
        mid_feats = in_feats // mid_factor
        self.softmax = nn.Softmax(dim=-1)
        self.att_conv = nn.Conv2d(in_feats, 1, kernel_size=1)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(in_feats, mid_feats, kernel_size=1),
            nn.LayerNorm([mid_feats, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_feats, in_feats, kernel_size=1)
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        x_in = x.view(b, 1, c, h*w)                                   # [b, 1, c, h*w]
        context_mask = self.att_conv(x).view(b, 1, -1)                # [b, 1, h*w]
        context_mask = self.softmax(context_mask).unsqueeze(-1)       # [b, 1, h*w, 1]
        context = torch.matmul(x_in, context_mask).view(b, -1, 1, 1)  # [b, c, 1, 1]
        out = x + self.channel_add_conv(context)
        return out

class ECA_Block(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, in_feats, gamma=2, b=1):
        super(ECA_Block, self).__init__()
        from math import log
        t = int(abs((log(in_feats, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class SCSE_Block(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch, ch//re,1),
                                 nn.ReLU(inplace = True),
                                 nn.Conv2d(ch//re, ch, 1),
                                 nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(ch, ch, 1),
                                 nn.Sigmoid())
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

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

class MultiResolutionFusion(nn.Module):
    def __init__(self, out_feats, *shapes):
        super().__init__()
        _, min_scale = min(shapes, key=lambda x: x[1])

        self.scale_factors = []
        for i, shape in enumerate(shapes):
            feat, scale= shape
            self.scale_factors.append(scale // min_scale)
            self.add_module("resolve{}".format(i),
                            nn.Conv2d(feat, out_feats, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, *xs):

        output = self.resolve0(xs[0])
        if self.scale_factors[0] != 1:
            output = nn.functional.interpolate(output, scale_factor=self.scale_factors[0], mode='bilinear', align_corners=True)

        for i, x in enumerate(xs[1:], 1): # the value for i starts from 1
            current_out = self.__getattr__("resolve{}".format(i))(x)
            if self.scale_factors[i] != 1:
                current_out = nn.functional.interpolate(current_out, scale_factor=self.scale_factors[i], mode='bilinear', align_corners=True)
            output += current_out

        return output

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

class MRF_Concat_6_2(nn.Module):
    def __init__(self, out_feats, *shapes):
        super().__init__()
        _, min_scale = min(shapes, key=lambda x: x[1])

        self.scale_factors = []
        self.concat_count = 0
        for shape in shapes:
            self.concat_count += shape[0]
            self.scale_factors.append(shape[1] // min_scale)
        
        self.out_block = nn.Conv2d(self.concat_count, out_feats, kernel_size=1, stride=1, padding=0, bias=False)

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

ATT_MODULE_DICT = {'gc': GC_Block, 'se': SE_Block, 'spa': SPA_Block, 'pp': PP_Block, 
                   'eca': ECA_Block, 'scse': SCSE_Block, 'ppc': PPC_Block, 'pdl': PDL_Block,
                   'sc': SC_Block, 'psc': PSC_Block, 'pca': PCA_Block, 'idt': Identity_ARM,
                   'lgc': LGC_Block, 'pgc': PGC_Block, 'gam': GAM_Block, 'agam': AGAM_Block,
                   'bgam': BGAM_Block, 'cgam': CGAM_Block, 'dgam': DGAM_Block, 'sam': SAM_Block}