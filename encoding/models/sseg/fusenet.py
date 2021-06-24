import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from ...nn import conv_block, up_conv, Attention_block, init_weights

__all__ = ['FuseNet', 'get_fuse', 'MFNet', 'get_mfnet']


class FuseNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, dropout=None):
        super(FuseNet, self).__init__()
        if dropout:
            self.drop_out = True

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.Unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256, n_conv=3)
        self.Conv4 = conv_block(ch_in=256, ch_out=512, n_conv=3)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024, n_conv=3)

        self.Dep_conv1 = conv_block(ch_in=1, ch_out=64)
        self.Dep_conv2 = conv_block(ch_in=64, ch_out=128)
        self.Dep_conv3 = conv_block(ch_in=128, ch_out=256, n_conv=3)
        self.Dep_conv4 = conv_block(ch_in=256, ch_out=512, n_conv=3)
        self.Dep_conv5 = conv_block(ch_in=512, ch_out=1024, n_conv=3)
        if self.drop_out:
            self.drop4 = nn.Dropout(p=dropout)
            self.drop5 = nn.Dropout(p=dropout)
            self.dep_drop4 = nn.Dropout(p=dropout)
            self.dep_drop5 = nn.Dropout(p=dropout)
            self.drop = nn.Dropout(p=dropout)
            self.decode_drop5 = nn.Dropout(p=dropout)
            self.decode_drop4 = nn.Dropout(p=dropout)
            self.decode_drop3 = nn.Dropout(p=dropout)

        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512, n_conv=3)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256, n_conv=3)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Up_conv1 = conv_block(ch_in=64, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, dep):
        # encoding Depth
        d1 = self.Dep_conv1(dep)

        d2, _ = self.Maxpool(d1)  # what if I change MaxPool to AvgPool
        d2 = self.Dep_conv2(d2)

        d3, _ = self.Maxpool(d2)
        d3 = self.Dep_conv3(d3)

        d4, _ = self.Maxpool(d3)
        if self.drop_out:
            d4 = self.dep_drop4(d4)
        d4 = self.Dep_conv4(d4)

        d5, _ = self.Maxpool(d4)
        if self.drop_out:
            d5 = self.dep_drop5(d5)
        d5 = self.Dep_conv5(d5)

        # encoding rgb
        x1 = self.Conv1(x)  # [64, h, w]
        x1 = x1 + d1  # fuse

        x2, idx2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)  # [128, h/2, w/2]
        x2 = x2 + d2  # fuse

        x3, idx3 = self.Maxpool(x2)  # [128, h/4, w/4]
        x3 = self.Conv3(x3)  # [256, h/4, w/4]
        x3 = x3 + d3  # fuse

        x4, idx4 = self.Maxpool(x3)  # [256, h/8, w/8] + Dropout?
        if self.drop_out:
            x4 = self.drop4(x4)
        x4 = self.Conv4(x4)  # [512, h/8, w/8]
        x4 = x4 + d4  # fuse

        x5, idx5 = self.Maxpool(x4)  # + Dropout?
        if self.drop_out:
            x5 = self.drop5(x5)
        x5 = self.Conv5(x5)  # [1024, h/16, w/16]
        x5 = x5 + d5  # fuse

        # original FuseNet: + MaxPool + Dropout + UnPool
        x, idx = self.Maxpool(x5)  # [ 1024, h/32, w/32]
        x = self.drop(x)

        # decoding + concat path
        dd5 = self.Unpool(x, idx)   # [1024, h/16, w/16]
        dd5 = self.Up_conv5(dd5)    # [512, h/16, w/16]  # + Dropout
        if self.drop_out:
            dd5 = self.decode_drop5(dd5)

        dd4 = self.Unpool(dd5, idx5)    # [512, h/8, w/8]
        dd4 = self.Up_conv4(dd4)        # [256, h/8, w/8]   # + Dropout
        if self.drop_out:
            dd4 = self.decode_drop4(dd4)

        dd3 = self.Unpool(dd4, idx4)  # [256, h/4, w/4]
        dd3 = self.Up_conv3(dd3)      # [128, h/4, w/4]  # + Dropout
        if self.drop_out:
            dd3 = self.decode_drop3(dd3)

        dd2 = self.Unpool(dd3, idx3)  # [128, h/2, w/2]
        dd2 = self.Up_conv2(dd2)      # [64, h/2, w/2]

        dd1 = self.Unpool(dd2, idx2)  # [64, h, w]
        dd1 = self.Up_conv1(dd1)      # [64, h, w]
        y = self.Conv_1x1(dd1)

        return y


def get_fuse(dataset='pascal_voc', backbone='resnet50', pretrained=False, dropout=None,
             root='../../encoding/models/pretrain', **kwargs):
    # infer number of classes
    from ...datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = FuseNet(img_ch=3, output_ch=datasets[dataset.lower()].NUM_CLASS, dropout=dropout)
    # init_weights(model)  # optional
    return model


class MFNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, inception=False):
        super(MFNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        dep_ch = [1, 32, 64, 128, 256, 512]
        rgb_ch = [3, 32, 64, 128, 256, 512]

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=rgb_ch[1])
        self.Conv2 = conv_block(ch_in=rgb_ch[1], ch_out=rgb_ch[2])
        self.Conv3 = conv_block(ch_in=rgb_ch[2], ch_out=rgb_ch[3])
        self.Conv4 = conv_block(ch_in=rgb_ch[3], ch_out=rgb_ch[4])
        self.Conv5 = conv_block(ch_in=rgb_ch[4], ch_out=rgb_ch[5])

        self.Dep_conv1 = conv_block(ch_in=1, ch_out=dep_ch[1])
        self.Dep_conv2 = conv_block(ch_in=dep_ch[1], ch_out=dep_ch[2])
        self.Dep_conv3 = conv_block(ch_in=dep_ch[2], ch_out=dep_ch[3])
        self.Dep_conv4 = conv_block(ch_in=dep_ch[3], ch_out=dep_ch[4])
        self.Dep_conv5 = conv_block(ch_in=dep_ch[4], ch_out=dep_ch[5])

        self.Up5 = up_conv(ch_in=rgb_ch[5] + dep_ch[5], ch_out=rgb_ch[4] + dep_ch[4])
        self.Up_conv5 = conv_block(ch_in=2*rgb_ch[4] + 2*dep_ch[4], ch_out=rgb_ch[4]+dep_ch[4])

        self.Up4 = up_conv(ch_in=rgb_ch[4] + dep_ch[4], ch_out=rgb_ch[3] + dep_ch[3])
        self.Up_conv4 = conv_block(ch_in=2*rgb_ch[3] + 2*dep_ch[3], ch_out=rgb_ch[3] + dep_ch[3])

        self.Up3 = up_conv(ch_in=rgb_ch[3] + dep_ch[3], ch_out=rgb_ch[2] + dep_ch[2])
        self.Up_conv3 = conv_block(ch_in=2*rgb_ch[2] + 2*dep_ch[2], ch_out=rgb_ch[2] + dep_ch[2])

        self.Up2 = up_conv(ch_in=rgb_ch[2] + dep_ch[2], ch_out=rgb_ch[1] + dep_ch[1])
        self.Up_conv2 = conv_block(ch_in=2*rgb_ch[1] + 2*dep_ch[1], ch_out=rgb_ch[1] + dep_ch[1])

        self.Conv_1x1 = nn.Conv2d(rgb_ch[1] + dep_ch[1], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, dep):
        # encoding Depth
        d1 = self.Dep_conv1(dep)  # [d1, h, w]

        d2, _ = self.Maxpool(d1)  # [d1, h/2, w/2]  what if I change MaxPool to AvgPool
        d2 = self.Dep_conv2(d2)   # [d2, h/2, w/2]

        d3, _ = self.Maxpool(d2)  # [d2, h/4, w/4]
        d3 = self.Dep_conv3(d3)   # [d3, h/4, w/4]

        d4, _ = self.Maxpool(d3)  # [d3, h/8, w/8]
        d4 = self.Dep_conv4(d4)   # [d4, h/8, w/8]

        d5, _ = self.Maxpool(d4)  # [d4, h/16, w/16]
        d5 = self.Dep_conv5(d5)   # [d5, h/16, w/16]

        # encoding rgb
        x1 = self.Conv1(x)            # [64, h, w]

        x2, idx2 = self.Maxpool(x1)  # [64, h/2, w/2]
        x2 = self.Conv2(x2)          # [128, h/2, w/2]

        x3, idx3 = self.Maxpool(x2)  # [128, h/4, w/4]
        x3 = self.Conv3(x3)          # [256, h/4, w/4]

        x4, idx4 = self.Maxpool(x3)  # [256, h/8, w/8]   # + Dropout?
        x4 = self.Conv4(x4)          # [512, h/8, w/8]

        x5, idx5 = self.Maxpool(x4)  # [512, h/16, w/16]  + Dropout?
        x5 = self.Conv5(x5)          # [1024, h/16, w/16]

        x = torch.cat((x5, d5), dim=1)  # concat RGB and depth feature

        # decoding + concat path
        y5 = self.Up5(x)                     # [512+d4, h/8, w/8]
        y5 = torch.cat((y5, x4, d4), dim=1)  # [1024+2d, h/8, w/8]
        y5 = self.Up_conv5(y5)               # [512+d4, h/8, w/8]

        y4 = self.Up4(y5)                    # [256+d3, h/4, w/4]
        y4 = torch.cat((y4, x3, d3), dim=1)  # [512+2d, h/4, w/4]
        y4 = self.Up_conv4(y4)               # [256+d3, h/4, w/4]

        y3 = self.Up3(y4)                    # [128+d2, h/2, w/2]
        y3 = torch.cat((y3, x2, d2), dim=1)  # [256+2d, h/2, w/2]
        y3 = self.Up_conv3(y3)               # [128+d2, h/2, w/2]

        y2 = self.Up2(y3)                    # [64+d1, h, w]
        y2 = torch.cat((y2, x1, d1), dim=1)  # [128+2d, h, w]
        y2 = self.Up_conv2(y2)               # [64+d1, h, w]

        y1 = self.Conv_1x1(y2)

        return y1


def get_mfnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
              root='../../encoding/models/pretrain', **kwargs):
    # infer number of classes
    from ...datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = MFNet(img_ch=3, output_ch=datasets[dataset.lower()].NUM_CLASS)
    # init_weights(model)  # optional
    return model
