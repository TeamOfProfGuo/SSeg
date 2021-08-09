
import torch
import torch.nn as nn
from .util import interpolate

class Centerpiece(nn.Module):
    def __init__(self, cp='psp', feats='x', cp_args={}):
        super().__init__()
        self.feats = feats
        self.feats_list = [64, 128, 256, 512]
        cp_dict = {'none': IDT_Module, 'psp': PSP_Module}
        for i in range(len(self.feats_list)):
            self.add_module('cp%d' % (i+1), cp_dict[cp](self.feats_list[i], **cp_args))
    
    def forward(self, feats):
        for i in range(len(self.feats_list)):
            k = '%s%d' % (self.feats, i+1)
            feats[k] = self.__getattr__('cp%d' % (i+1))(feats[k])
        return feats

class IDT_Module(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class PSP_Module(nn.Module):
    def __init__(self, in_feats, size=(1, 3, 5, 7), up_mode='nearest', keep_feats=True):
        super().__init__()
        self.size = size
        self.up_mode = up_mode

        if keep_feats:
            out_feats = in_feats
            self.rf_conv = nn.Sequential(
                nn.Conv2d(in_feats, out_feats//2, kernel_size=1),
                nn.BatchNorm2d(out_feats//2),
                nn.ReLU(inplace=True)
            )
        else:
            out_feats = 2 * in_feats
            self.rf_conv = nn.Identity()

        for s in size:
            self.add_module('pool%d' % s, nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_feats, out_feats//8, kernel_size=1),
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