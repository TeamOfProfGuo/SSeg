
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['ConvBnAct', 'ResidualBasicBlock', 'ResidualDecBlock', 'NBC_Block', 'LU_Unit', 'interpolate', 'up_block', 'out_block']

class ConvBnAct(nn.Sequential):
    def __init__(self, in_feats, out_feats, kernel=3, stride=1, pad=1, bias=False, conv_args = {},
                 norm_layer=nn.BatchNorm2d, act=True, act_layer=nn.ReLU(inplace=True)):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_feats, out_feats, kernel_size=kernel, stride=stride,
                                            padding=pad, bias=bias, **conv_args))
        self.add_module('bn', norm_layer(out_feats))
        self.add_module('act', act_layer if act else nn.Identity())

class ResidualBasicBlock(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.conv_unit = nn.Sequential(
            ConvBnAct(in_feats, in_feats),
            ConvBnAct(in_feats, in_feats, act=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_unit(x)
        return self.relu(x + out)

class ResidualDecBlock(nn.Module):
    def __init__(self, in_feats, out_feats, upsample=False, lu='ori'):
        super().__init__()
        if upsample:
            self.up1 = LearnedUpUnit(out_feats) if lu == 'ori' else LU_Unit('lurp', out_feats)
            self.up2 = LearnedUpUnit(out_feats) if lu == 'ori' else LU_Unit('lurp', out_feats)
        else:
            self.up1 = nn.Identity()
            self.up2 = nn.Identity()
        self.dec_conv = ConvBnAct(in_feats, out_feats)
        self.out_conv = ConvBnAct(out_feats, out_feats, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        idt = self.dec_conv(x)
        out = self.out_conv(idt)
        idt = self.up1(idt)
        out = self.up2(out)
        return self.relu(idt + out)

class NBC_Up(nn.Sequential):
    def __init__(self, in_feats):
        super().__init__()
        dec_feats = in_feats // 2
        self.add_module('dec_conv', ConvBnAct(in_feats, dec_feats))
        self.add_module('nbc_block', NBC_Block(dec_feats, nbc_num=3))
        self.add_module('lu_unit', LU_Unit('lurp', dec_feats))

class NBC_Block(nn.Sequential):
    def __init__(self, in_feats, nbc_num=3):
        super().__init__()   
        for i in range(nbc_num):
            self.add_module('nbc%d' % i, NonBottleneck1D(in_feats, in_feats))

class NonBottleneck1D(nn.Module):
    """
    ERFNet-Block
    Paper:
    http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf
    Implementation from:
    https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=None, dilation=1, norm_layer=None,
                 activation=nn.ReLU(inplace=True), residual_only=False):
        super().__init__()
        
        dropprob = 0
        self.conv3x1_1 = nn.Conv2d(inplanes, planes, (3, 1),
                                   stride=(stride, 1), padding=(1, 0),
                                   bias=True)
        self.conv1x3_1 = nn.Conv2d(planes, planes, (1, 3),
                                   stride=(1, stride), padding=(0, 1),
                                   bias=True)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-03)
        self.act = activation
        self.conv3x1_2 = nn.Conv2d(planes, planes, (3, 1),
                                   padding=(1 * dilation, 0), bias=True,
                                   dilation=(dilation, 1))
        self.conv1x3_2 = nn.Conv2d(planes, planes, (1, 3),
                                   padding=(0, 1 * dilation), bias=True,
                                   dilation=(1, dilation))
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)
        self.downsample = downsample
        self.stride = stride
        self.residual_only = residual_only

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.act(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.act(output)

        output = self.conv3x1_2(output)
        output = self.act(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        if self.downsample is None:
            identity = input
        else:
            identity = self.downsample(input)

        if self.residual_only:
            return output

        return self.act(output + identity)

class LU_Unit(nn.Module):
    def __init__(self, mode='lurp', in_feats=None):
        super().__init__()

        if mode == 'lurp':
            up_params_flag = True
            self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
            self.conv = nn.Conv2d(in_feats, in_feats, groups=in_feats, kernel_size=3, padding=0)
        elif mode == 'luzp':
            up_params_flag = True
            self.pad = nn.Identity()
            self.conv = nn.Conv2d(in_feats, in_feats, groups=in_feats, kernel_size=3, padding=1)
        else:
            up_params_flag = False
            self.mode = mode
            self.pad = nn.Identity()
            self.conv = nn.Identity()

        if up_params_flag:
            # interpolate mode
            self.mode = 'nearest'

            # kernel that mimics bilinear interpolation
            w = torch.tensor([[[
                [0.0625, 0.1250, 0.0625],
                [0.1250, 0.2500, 0.1250],
                [0.0625, 0.1250, 0.0625]
            ]]])

            self.conv.weight = torch.nn.Parameter(torch.cat([w] * in_feats))
            # set bias to zero
            with torch.no_grad():
                self.conv.bias.zero_()

    def forward(self, x):
        _, _, h, w = x.size()
        x = interpolate(x, (2*h, 2*w), mode=self.mode)
        x = self.pad(x)
        x = self.conv(x)
        return x


def interpolate(x, size, mode = 'bilinear'):
    if mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
        return F.interpolate(x, size=size, mode=mode, align_corners=True)
    else:
        return F.interpolate(x, size=size, mode=mode)

def up_block(feats, module='cbr'):
    if module == 'cbr':
        return nn.Sequential(
            CBR(2*feats, feats),
            LearnedUpUnit(feats)
            # LU_Unit('lurp', feats)
        )
    elif module == 'bb':
        return nn.Sequential(
            BasicBlock(2*feats, 2*feats),
            BasicBlock(2*feats, feats, upsample=True)
        )
    elif module in ('rbb', 'rbb6', 'rbb7'):
        # # Ver 1
        # return nn.Sequential(
        #     ResidualBasicBlock(2*feats),
        #     ResidualDecBlock(2*feats, feats, upsample=True)
        # )
        # Ver 2
        return nn.Sequential(
            ResidualBasicBlock(2*feats),
            ResidualBasicBlock(2*feats),
            ResidualDecBlock(2*feats, feats, upsample=True)
        )
    elif module == 'nbc':
        return NBC_Up(2*feats)
    else:
        raise NotImplementedError('Invalid decoder module: %s.' % module)

def out_block(in_feats, mid_feats, n_classes, module='cbr'):
    if module == 'cbr':
        return nn.Sequential(
            CBR(in_feats, mid_feats),
            CBR(mid_feats, mid_feats),
            nn.Conv2d(mid_feats, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
    elif module == 'bb':
        return nn.Sequential(
            BasicBlock(in_feats, mid_feats, upsample=True),
            BasicBlock(mid_feats, mid_feats),
            nn.Conv2d(mid_feats, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
    elif module == 'rbb6':
        # Ver 1
        return nn.Sequential(
            ResidualDecBlock(in_feats, mid_feats, upsample=True),
            ResidualBasicBlock(mid_feats),
            nn.Conv2d(mid_feats, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
    elif module in ('rbb', 'rbb7'):
        # # Ver 1
        # return nn.Sequential(
        #     ResidualDecBlock(in_feats, mid_feats, upsample=True),
        #     ResidualBasicBlock(mid_feats),
        #     nn.Conv2d(mid_feats, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # )
        # Ver 2
        return nn.Sequential(
            nn.Conv2d(in_feats, n_classes, kernel_size=1, stride=1, padding=0, bias=True),
            LearnedUpUnit(n_classes),
            LearnedUpUnit(n_classes)
            # LU_Unit('lurp', n_classes),
            # LU_Unit('lurp', n_classes)
        )
        # # Ver 3
        # return nn.Sequential(
        #     ResidualBasicBlock(in_feats),
        #     nn.Conv2d(in_feats, n_classes, kernel_size=1, stride=1, padding=0, bias=True),
        #     LU_Unit('lurp', n_classes),
        #     LU_Unit('lurp', n_classes)
        # )
    elif module == 'nbc':
        return nn.Sequential(
            nn.Conv2d(in_feats, n_classes, kernel_size=3, padding=1),
            LU_Unit('lurp', n_classes),
            LU_Unit('lurp', n_classes)
        )
    else:
        raise NotImplementedError('Invalid decoder module: %s.' % module)

class LearnedUpUnit(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.dep_conv = nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1, padding=1, groups=in_feats, bias=False)

    def forward(self, x):
        x = self.up(x)
        x = self.dep_conv(x)
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