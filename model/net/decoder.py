
import torch
import torch.nn as nn

from .util import *
from .apnb import APNB
from .fuse import FUSE_MODULE_DICT

class Decoder(nn.Module):
    def __init__(self, n_classes, feats='x', aux=False, final_fuse='none', lf_args={}, final_args={}):
        super().__init__()

        self.aux = aux
        self.feats = feats
        self.final_fuse = final_fuse and aux
        
        decoder_feats = [256, 128, 64]

        # Refine Blocks
        for i in range(len(decoder_feats)):
            self.add_module('refine%d' % i,
                Level_Fuse_Module(in_feats=decoder_feats[i], **lf_args)
            )

        # Upsample Blocks
        for i in range(len(decoder_feats)):
            self.add_module('up%d' % i, 
                IRB_Up_Block(decoder_feats[i], aux=aux)
            )
        
        # Aux loss
        if aux:
            for i in range(len(decoder_feats)):
                self.add_module('aux%d' % i, 
                    nn.Conv2d(decoder_feats[i], n_classes, kernel_size=1, stride=1, padding=0, bias=True),
                )
        
        # Final fusion
        final_dict = {
            'none': nn.Identity,
            'apnb': APNB,
            'sam': SimAM_Block
        }
        self.out_conv = nn.Sequential(
            final_dict[final_fuse](**final_args),
            nn.Conv2d(min(decoder_feats), n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.out_up = nn.Sequential(
            LearnedUpUnit(n_classes),
            LearnedUpUnit(n_classes)
        )

    def forward(self, in_feats):
        f1 = in_feats['%s1' % self.feats]
        f2 = in_feats['%s2' % self.feats]
        f3 = in_feats['%s3' % self.feats]
        f4 = in_feats['%s4' % self.feats]

        if self.aux:
            # Level Fuse
            feats, aux0 = self.up0(f4)
            feats = self.refine0(feats, f3)
            feats, aux1 = self.up1(feats)
            feats = self.refine1(feats, f2)
            feats, aux2 = self.up2(feats)
            feats = self.refine2(feats, f1)
            # Output
            aux3 = self.out_conv(feats)
            out_feats = [self.out_up(aux3), aux3, self.aux2(aux2), self.aux1(aux1), self.aux0(aux0)]
            return out_feats
        else:
            feats = self.refine0(self.up0(f4), f3)
            feats = self.refine1(self.up1(feats), f2)
            feats = self.refine2(self.up2(feats), f1)
            return [self.out_conv(feats)]

class Level_Fuse_Module(nn.Module):
    def __init__(self, in_feats, conv_flag=(True, False), lf_bb='rbb[2->2]', fuse_args={}, fuse_module='fuse'):
        super().__init__()
        self.conv_flag = conv_flag
        self.fuse = FUSE_MODULE_DICT[fuse_module](in_feats, **fuse_args)
        self.rfb0 = customized_module(lf_bb, in_feats) if conv_flag[0] else nn.Identity()
        self.rfb1 = customized_module(lf_bb, in_feats) if conv_flag[1] else nn.Identity()
    
    def forward(self, y, x):
        x = self.rfb0(x)    # Refine feats from backbone
        out, _, _ = self.fuse(y, x)
        return self.rfb1(out)

class IRB_Up_Block(nn.Module):
    def __init__(self, in_feats, aux=False):
        super().__init__()
        self.aux = aux
        self.conv_unit = nn.Sequential(
            IRB_Block(2*in_feats, 2*in_feats),
            IRB_Block(2*in_feats, 2*in_feats),
            IRB_Block(2*in_feats, in_feats)
        )
        self.up_unit = LearnedUpUnit(in_feats)

    def forward(self, x):
        feats = self.conv_unit(x)
        if self.aux:
            return self.up_unit(feats), feats
        else:
            return self.up_unit(feats)

class FF_Block(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.final_fuse = nn.Conv2d(4*n_classes, n_classes, kernel_size=1, padding=0, groups=n_classes, bias=True)
        self.final_fuse.weight.data = init_conv(n_classes, 4, 1, mode='b')
        
    def forward(self, *feats):
        feats_pyramid = []
        b, c, h, w = feats[0].size()
        for feats in tuple(feats):
            feats_pyramid.append(interpolate(feats, size=(h, w), mode='nearest'))
        out = torch.cat(tuple(feats_pyramid), dim=-2).reshape(b, 4*c, h, w)
        return [self.final_fuse(out)] + feats[1:]