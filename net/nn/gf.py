
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..utils import interpolate

__all__ = ['GF_Module', 'RC_Block']

class GF_Module(nn.Module):
    def __init__(self, feats1: int, feats2=None, info=None):
        super().__init__()

        if feats2 is None:
            feats2 = feats1
        self.pp1 = nn.Identity()
        self.pp2 = nn.Identity()
        #     # nn.Sequential(
        #     #     nn.Conv2d(feats2, feats1, kernel_size=1, bias=False),
        #     #     nn.BatchNorm2d(feats1),
        #     #     nn.ReLU(inplace=True)
        #     # )
        # else:
        #     psp_args = {'size': (1, 3, 5, 7), 'up_mode': 'nearest'}
        #     self.pp1 = PSP_Block(feats1, feats1, **psp_args)
        #     self.pp2 = PSP_Block(feats2, feats1, **psp_args)

        # psp_args = {'size': (1, 3, 5, 7), 'up_mode': 'nearest'}
        # self.pp1 = PSP_Block(feats1, feats1, **psp_args)
        # self.pp2 = PSP_Block(feats2, feats1, **psp_args)

        # self.pp1 = nn.Conv2d(feats1, feats1, kernel_size=1)
        # self.pp2 = nn.Conv2d(feats2, feats1, kernel_size=1)
        
        gau_up_mode = 'bilinear'
        self.gau = GAU_Block(feats1, gau_up_mode)

    
    def forward(self, f1, f2):
        f0 = f2.clone()
        f1 = self.pp1(f1)
        f2 = self.pp2(f2)
        return self.gau(f1, f2), f0

class GAU_Block(nn.Module):
    def __init__(self, in_feats: int, up_mode='bilinear'):
        super().__init__()
        self.up_mode = up_mode
        self.x_conv = nn.Sequential(nn.Conv2d(in_feats, in_feats, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_feats))
        self.y_conv = nn.Sequential(nn.Conv2d(in_feats, in_feats, kernel_size=1, padding=0, bias=False),
                                    nn.BatchNorm2d(in_feats),
                                    nn.ReLU(inplace=True))

    def forward(self, y, x):
        _, _, h, w = y.size()
        x = self.x_conv(interpolate(x, (h, w), self.up_mode))
        w = self.y_conv(F.adaptive_avg_pool2d(y, 1))
        return y + w * x

class PSP_Block(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, size=(1, 3, 5, 7), up_mode='nearest'):
        super().__init__()
        self.size = size
        self.up_mode = up_mode
        self.rf_conv = nn.Sequential(
            nn.Conv2d(in_feats, out_feats//2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_feats//2),
            nn.ReLU(inplace=True)
        )
        for s in size:
            self.add_module('pool%d' % s, nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_feats, out_feats//8, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_feats//8),
                nn.ReLU(inplace=True)
            ))
        
    def forward(self, x):
        _, _, h, w = x.size()
        out = [self.rf_conv(x)]
        for s in self.size:
            feats = self.__getattr__('pool%d' % s)(x)
            out.append(interpolate(feats, (h, w), self.up_mode))
        return torch.cat(tuple(out), dim=1)

class RC_Block(nn.Module):
    def __init__(self, feats: int, unit_num=2):
        super().__init__()

        self.unit_num = unit_num
        for i in range(unit_num):
            self.add_module('conv%d' % i, nn.Sequential(
                nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(feats),
                nn.ReLU(inplace=True),
                nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(feats)
            ))

    def forward(self, x):
        for i in range(self.unit_num):
            y = x.clone()
            x = self.__getattr__('conv%d' % i)(x)
            x = x + y
        return x

class CBR(nn.Module):
    def __init__(self, in_feats, out_feats, k=2):
        super().__init__()
        self.mid_cbr = nn.Sequential(nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(in_feats), nn.ReLU(inplace=True)) if k != 1 else None
        self.out_cbr = nn.Sequential(nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_feats), nn.ReLU(inplace=True))
        
    def forward(self, x):
        if self.mid_cbr is not None:
            x = self.mid_cbr(x)
        return self.out_cbr(x)