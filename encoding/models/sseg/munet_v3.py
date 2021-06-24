import torch
import os
import torch.nn as nn
from torchvision import models
from torch.nn.functional import interpolate

__all__ = ['MUNet', 'get_munet']


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=nn.BatchNorm2d):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MUNet(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, n_classes=21, backbone='mobilenet_v2', pretrained=True, root='./encoding/models/pretrain', ):
        super(MUNet, self).__init__()

        base = models.mobilenet_v2(pretrained=False)
        self.features = base.features
        if pretrained:
            f_path = os.path.abspath(os.path.join(root, 'mobilenet_v2-b0353104.pth'))
            if not os.path.exists(f_path):
                raise FileNotFoundError('the pretrained model can not be found')
            base.load_state_dict(torch.load(f_path), strict=False)

        # decoder
        self.dconv4 = nn.ConvTranspose2d(in_channels=1280, out_channels=96, kernel_size=4, padding=1, stride=2)
        self.invres4 = InvertedResidual(inp=192, oup=96, stride=1, expand_ratio=6)

        self.dconv3 = nn.ConvTranspose2d(in_channels=96, out_channels=32, kernel_size=4, padding=1, stride=2)
        self.invres3 = InvertedResidual(inp=64, oup=32, stride=1, expand_ratio=6)

        self.dconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=24, kernel_size=4, padding=1, stride=2)
        self.invres2 = InvertedResidual(inp=48, oup=24, stride=1, expand_ratio=6)

        self.dconv1 = nn.ConvTranspose2d(in_channels=24, out_channels=16, kernel_size=4, padding=1, stride=2)
        self.invres1 = InvertedResidual(inp=32, oup=32, stride=1, expand_ratio=6)

        self.tp_conv = nn.ConvTranspose2d(32, n_classes, kernel_size=2,stride=2, padding=0)

        # Classifier
        # self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, kernel_size=2,stride=2, padding=0)

    def forward(self, x):
        _, _, h, w = x.size()

        # Encoder
        e0 = self.features[0](x)  # [B, 32, h/2, w/2]

        e1 = self.features[1](e0)  # [B, 16, h/2, w/2]  240

        e2 = self.features[2](e1)  # [B, 24, h/4, w/8]  120
        e2 = self.features[3](e2)  # [B, 24, h/4, w/4]  120

        e3 = self.features[4](e2)  # [B, 32, h/8, w/8] 60
        e3 = self.features[5](e3)  # [B, 32, h/8, w/8] 60
        e3 = self.features[6](e3)  # [B, 32, h/8, w/8] 60

        e4 = self.features[7](e3)  # [B, 64, h/16, w/16] 30
        e4 = self.features[8](e4)  # [B, 64, h/16, w/16] 30
        e4 = self.features[9](e4)  # [B, 64, h/16, w/16] 30
        e4 = self.features[10](e4)  # [B, 64, h/16, w/16] 30

        e4 = self.features[11](e4)  # [B, 96, h/16, w/16] 30
        e4 = self.features[12](e4)  # [B, 96, h/16, w/16] 30
        e4 = self.features[13](e4)  # [B, 96, h/16, w/16] 30

        e5 = self.features[14](e4)  # [B, 160, h/32, w/32]  15
        e5 = self.features[15](e5)  # [B, 160, h/32, w/32]  15
        e5 = self.features[16](e5)  # [B, 160, h/32, w/32]  15

        e5 = self.features[17](e5)  # [B, 320, h/32, w/32]  15
        e5 = self.features[18](e5)  # [B, 1280, h/32, w/32] 15

        d4 = self.dconv4(e5)  # [B, 96, h/16, w/16]
        d4 = torch.cat((d4, e4), dim=1)  # [B, 192, h/16, w/16]
        d4 = self.invres4(d4)  # [B, 96, h/16, w/16]

        d3 = self.dconv3(d4)  # [B, 32, h/8, w/8]
        d3 = torch.cat((d3, e3), dim=1)  # [B, 64, h/8, w/8]
        d3 = self.invres3(d3)  # [B, 32, h/8, w/8]

        d2 = self.dconv2(d3)  # [B, 24, h/4, w/4]
        d2 = torch.cat((d2, e2), dim=1)  # [B, 48, h/4, w/4]
        d2 = self.invres2(d2)  # [B, 24, h/4, w/4]

        d1 = self.dconv1(d2)  # [B, 16, h/2, w/2]
        d1 = torch.cat((d1, e1), dim=1)  # [B, 32, h/2, w/2]
        d1 = self.invres1(d1)  # [B, 16, h/2, w/2]

        y = self.tp_conv(d1)   # [B, n_class, h, w]

        return y


def get_munet(dataset='nyud', backbone='mobilenet_v2', pretrained=True, root='./encoding/models/pretrain',):
    from ...datasets import datasets
    model = MUNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, pretrained=pretrained, root=root)
    return model