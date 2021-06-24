
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

__all__ = ['init_weights', 'conv_block', 'up_conv', 'single_conv', 'Attention_block', ]


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, n_conv=2):
        super(conv_block, self).__init__()
        self.n_conv=n_conv
        modules = [nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                   nn.BatchNorm2d(ch_out),
                   nn.ReLU(inplace=True)]
        for i in range(n_conv-1):
            modules.extend([nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.BatchNorm2d(ch_out),
                            nn.ReLU(inplace=True)])

        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x


# class up_conv(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(up_conv, self).__init__()
#         self.up = nn.MaxUnpool2d(kernel_size=2, stride=2)
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True))
#
#     def forward(self, x, idx):
#         x = self.conv(x)
#         x = self.up(x, idx)
#         return x


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)   # no bias
        psi = self.psi(psi)

        return x * psi
