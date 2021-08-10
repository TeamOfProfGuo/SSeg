
import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import init_conv

class Fuse_Module(nn.Module):
    def __init__(self, in_feats, fuse_setting={}, att_module='idt', att_setting={}):
        super().__init__()
        module_dict = {
            'idt': IDT_Block,
            'se': SE_Block,
            'pdl': PDL_Block
        }
        self.att_module = att_module
        self.pre1 = module_dict[att_module](in_feats, **att_setting)
        self.pre2 = module_dict[att_module](in_feats, **att_setting)
        self.gcgf = General_Fuse_Block(in_feats, **fuse_setting)
    
    def forward(self, x, y):
        if self.att_module != 'idt':
            x = self.pre1(x)
            y = self.pre2(y)
        return self.gcgf(x, y), x, y

class General_Fuse_Block(nn.Module):
    def __init__(self, in_feats, pre_bn=False, merge='gcgf', init=(False, True), civ=1):
        super().__init__()
        merge_dict = {
            'gcgf': nn.Conv2d(2*in_feats, in_feats, kernel_size=1, padding=0, groups=in_feats, bias=True),
            'add': Add_Merge(in_feats),
            'cc3': CC3_Merge(in_feats),
            'la': LA_Merge(in_feats)
        }
        if pre_bn:
            self.pre_bn1 = nn.BatchNorm2d(in_feats)
            self.pre_bn2 = nn.BatchNorm2d(in_feats)
        self.pre_bn = pre_bn
        self.feats = in_feats
        self.merge_mode = merge
        self.merge = merge_dict[merge]
        self._init_weights(init, civ)
        
    def forward(self, x, y):
        b, c, h, w = x.size()
        if self.pre_bn:
            x = self.pre_bn1(x)
            y = self.pre_bn2(y)
        if self.merge_mode != 'gcgf':
            return self.merge(x, y)
        feats = torch.cat((x, y), dim=-2).reshape(b, 2*c, h, w)   # [b, c, 2h, w] => [b, 2c, h, w]
        return self.merge(feats)
            
    def _init_weights(self, init, civ):
        if init[0] and self.pre_bn:
            self.pre_bn1.weight.data.fill_(1)
            self.pre_bn2.weight.data.fill_(1)
            self.pre_bn1.bias.data.zero_()
            self.pre_bn2.bias.data.zero_()
        if init[1] and isinstance(self.merge, nn.Conv2d):
            if civ == -1:
                self.merge.weight.data = init_conv(self.feats, 2, 1, 'b')
            else:
                self.merge.weight.data.fill_(civ)

class Merge_Module(nn.Module):
    def __init__(self, in_feats, fuse_setting={}, att_module='idt', att_setting={}):
        super().__init__()
        module_dict = {
            'idt': IDT_Block,
            'se': SE_Block,
            'pdl': PDL_Block
        }
        self.att_module = att_module
        self.pre1 = module_dict[att_module](in_feats, **att_setting)
        self.pre2 = module_dict[att_module](in_feats, **att_setting)
        self.gcgf = General_Merge_Block(in_feats, **fuse_setting)
    
    def forward(self, m, x, y):
        if self.att_module != 'idt':
            x = self.pre1(x)
            y = self.pre2(y)
        return self.gcgf(m, x, y), x, y

class General_Merge_Block(nn.Module):
    def __init__(self, in_feats, pre_bn=False, merge='gcgf', init=(False, True), civ=1):
        super().__init__()
        if pre_bn:
            self.pre_bn1 = nn.BatchNorm2d(in_feats)
            self.pre_bn2 = nn.BatchNorm2d(in_feats)
        self.pre_bn = pre_bn
        self.feats = in_feats
        self.merge = nn.Conv2d(3*in_feats, in_feats, kernel_size=1, padding=0, groups=in_feats, bias=True)
        self._init_weights(init, civ)
        
    def forward(self, m, x, y):
        b, c, h, w = x.size()
        if self.pre_bn:
            x = self.pre_bn1(x)
            y = self.pre_bn2(y)
        feats = torch.cat((m, x, y), dim=-2).reshape(b, 3*c, h, w)
        return self.merge(feats)
            
    def _init_weights(self, init, civ):
        if init[0] and self.pre_bn:
            self.pre_bn1.weight.data.fill_(1)
            self.pre_bn2.weight.data.fill_(1)
            self.pre_bn1.bias.data.zero_()
            self.pre_bn2.bias.data.zero_()
        if init[1] and isinstance(self.merge, nn.Conv2d):
            if civ == -1:
                self.merge.weight.data = init_conv(self.feats, 3, 1, 'b')
            else:
                self.merge.weight.data.fill_(civ)

# ARM Blocks

class SE_Block(nn.Module):
    def __init__(self, in_feats, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_feats, in_feats // r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_feats // r, in_feats, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(F.adaptive_avg_pool2d(x, 1))
        return w * x
    
class PDL_Block(nn.Module):
    def __init__(self, in_feats, pp_layer=4, descriptor=8, mid_feats=16):
        super().__init__()
        self.layer_size = pp_layer                  # l: pyramid layer num
        self.feats_size = (4 ** pp_layer - 1) // 3  # f: feats for descritor
        self.descriptor = descriptor                # d: descriptor num (for one channel)

        self.des = nn.Conv2d(self.feats_size, descriptor, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(descriptor * in_feats, mid_feats, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_feats, in_feats),
            nn.Sigmoid()
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
        return w * x

class IDT_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

# Merge Blocks

class Add_Merge(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, x, y):
        return x+y

class LA_Merge(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lamb = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, y):
        return x + self.lamb * y

class CC3_Merge(nn.Module):
    def __init__(self, in_feats, *args, **kwargs):
        super().__init__()
        self.cc_block = nn.Sequential(
            nn.Conv2d(2*in_feats, in_feats, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, y):
        return self.cc_block(torch.cat((x, y), dim=1))