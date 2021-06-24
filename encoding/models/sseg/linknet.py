import torch
import os
import torch.nn as nn
from torch.nn import init
from .base import BaseNet
from ..backbone import *
from torch.autograd import Variable
from torchvision.models import resnet
from ...nn import conv_block, up_conv, Attention_block, init_weights

__all__ = ['LinkNet', 'get_linknet', 'AttLinkNet', 'get_attlinknet']


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, kernel_size=1, stride=1, padding=0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x


class LinkNet(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, root='./encoding/models/pretrain', ):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(LinkNet, self).__init__()

        # base = resnet.resnet18(pretrained=True)
        base = resnet.resnet18(pretrained=False)
        if pretrained:
            if backbone=='resnet18':
                f_path = os.path.abspath(os.path.join(root, 'resnet18-5c106cde.pth'))
            if not os.path.exists(f_path):
                raise FileNotFoundError('the pretrained model can not be found')
            base.load_state_dict(torch.load(f_path), strict=False)

        self.in_block = nn.Sequential(   # [B, 64, h/4, w/4]
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1    # [B, 64, h/4, w/4]
        self.encoder2 = base.layer2    # [B, 128, h/8, w/8]
        self.encoder3 = base.layer3    # [B, 256, h/16, w/16]
        self.encoder4 = base.layer4    # [B, 512, h/32, w/32]

        self.decoder1 = Decoder(64, 64, kernel_size=3, stride=1, padding=1,)
        self.decoder2 = Decoder(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = Decoder(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder4 = Decoder(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, kernel_size=2,stride=2, padding=0)

    def forward(self, x):
        # Initial block
        x = self.in_block(x)    # [B, 64, h/4, w/4]   120

        # Encoder blocks
        e1 = self.encoder1(x)   # [B, 64, h/4, w/4]  120
        e2 = self.encoder2(e1)  # [B, 128, h/8, w/8]  60
        e3 = self.encoder3(e2)  # [B, 256, h/16, w/16]  30
        e4 = self.encoder4(e3)  # [B, 512, h/32, w/32]  15

        # Decoder blocks
        #d4 = e3 + self.decoder4(e4)
        d4 = e3 + self.decoder4(e4)  # [B, 256, h/16, w/16]  30
        d3 = e2 + self.decoder3(d4)  # [B, 128, h/8, w/8]  60
        d2 = e1 + self.decoder2(d3)  # [B, 64, h/4, w/4]  120
        d1 = x + self.decoder1(d2)   # [B, 64, h/4, w/4]  120

        # Classifier
        y = self.tp_conv1(d1)  # [B, 32, h/2, w/2] 240
        y = self.conv2(y)      # [B, 32, h/2, w/2] 240
        y = self.tp_conv2(y)   # [B, nclass, h, w] 480

        return y


def get_linknet(dataset='nyud', backbone='resnet34', root='./encoding/models/pretrain',):
    from ...datasets import datasets
    model = LinkNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root)
    return model


class AttLinkNet(nn.Module):
    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, root='./encoding/models/pretrain', ):
        super(AttLinkNet, self).__init__()

        base = resnet.resnet18(pretrained=False)
        if pretrained:
            if backbone == 'resnet18':
                f_path = os.path.abspath(os.path.join(root, 'resnet18-5c106cde.pth'))
            if not os.path.exists(f_path):
                raise FileNotFoundError('the pretrained model can not be found')
            base.load_state_dict(torch.load(f_path), strict=False)

        self.in_block0 = nn.Sequential(base.conv1,
                                       base.bn1,
                                       base.relu,)  # [B, 64, h/2, w/2]
        self.in_block1 = base.maxpool # [B, 64, h/4, w/4]

        self.encoder1 = base.layer1   # [B, 64, h/4, w/4]
        self.encoder2 = base.layer2   # [B, 128, h/8, w/8]
        self.encoder3 = base.layer3   # [B, 256, h/16, w/16]
        self.encoder4 = base.layer4   # [B, 512, h/32, w/32]

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True), )
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        # encoding path
        x0 = self.in_block0(x)   # [B, 64, h/2, w/2]
        x1 = self.in_block1(x0)  # [B, 64, h/4, w/4]
        x1 = self.encoder1(x1)   # [B, 64, h/4, w/4]

        x2 = self.encoder2(x1)  # [B, 128, h/8, w/8]
        x3 = self.encoder3(x2)  # [B, 256, h/16, w/16]
        x4 = self.encoder4(x3)  # [B, 512, h/32, w/32]

        # decoding + concat path
        d4 = self.Up4(x4)           # [B, 256, h/16, w/16]
        x3 = self.Att4(g=d4, x=x3)  # [B, 256, h/16, w/16]
        d4 = torch.cat((x3, d4), dim=1)  # [B, 512, h/16, w/16]
        d4 = self.Up_conv4(d4)      # [B, 256, h/16, w/16]

        d3 = self.Up3(d4)                # [B, 128, h/8, w/8]
        x2 = self.Att3(g=d3, x=x2)       # [B, 128, h/8, w/8]
        d3 = torch.cat((x2, d3), dim=1)  # [B, 256, h/8, w/8]
        d3 = self.Up_conv3(d3)           # [B, 128, h/8, w/8]

        d2 = self.Up2(d3)                # [B, 64, h/4, w/4]
        x1 = self.Att2(g=d2, x=x1)       # [B, 64, h/4, w/4]
        d2 = torch.cat((x1, d2), dim=1)  # [B, 128, h/4, w/4]
        d2 = self.Up_conv2(d2)           # [B, 64, h/4, w/4]

        # Classifier
        y = self.tp_conv1(d2)  # [B, 32, h/2, w/2] 240
        y = self.conv2(y)  # [B, 32, h/2, w/2] 240
        y = self.tp_conv2(y)  # [B, nclass, h, w] 480

        return y


def get_attlinknet(dataset='nyud', backbone='resnet18', pretrained=True, root='./encoding/models/pretrain',):
    from ...datasets import datasets
    model = AttLinkNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, pretrained=pretrained, root=root)
    return model
