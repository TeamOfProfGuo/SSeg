import torch
import torch.nn as nn
import torchvision
from copy import deepcopy

layers = [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]


class FuseNet(nn.Module):
    def __init__(self, nclass, layers, init_weights=True, drop_out=0.5):
        super(FuseNet, self).__init__()
        # rgb encoder
        enc = list(torchvision.models.vgg16_bn(pretrained=init_weights).features.children())
        self.layer0 = nn.Sequential(*deepcopy(enc[0:6]))
        self.layer1 = nn.Sequential(*deepcopy(enc[7:13]))
        self.layer2 = nn.Sequential(*deepcopy(enc[14:23]))
        self.layer3 = nn.Sequential(*deepcopy(enc[24:33]))
        self.layer4 = nn.Sequential(*deepcopy(enc[34:43]))

        # depth encoder: input 1 channel
        d_enc = list(torchvision.models.vgg16_bn(pretrained=init_weights).features.children())
        in_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.dlayer0 = nn.Sequential(*([in_conv] + deepcopy(d_enc[1:6])))
        self.dlayer1 = nn.Sequential(*deepcopy(d_enc[7:13]))
        self.dlayer2 = nn.Sequential(*deepcopy(d_enc[14:23]))
        self.dlayer3 = nn.Sequential(*deepcopy(d_enc[24:33]))
        self.dlayer4 = nn.Sequential(*deepcopy(d_enc[34:43]))

        self.dec4 = make_layers(layers[4], 512, 512)  # F:[512,512],[512,512],[512,512]
        self.dec3 = make_layers(layers[3], 512, 256)  # F:[512,512],[512,512],[512,256]
        self.dec2 = make_layers(layers[2], 256, 128)  # F:[256,256],[256,256],[256,128]
        self.dec1 = make_layers(layers[1], 128, 64)  # F:[128,128],[128,64]
        self.dec0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, nclass, kernel_size=3, padding=1),
        )
        # pool layer has no params, so we can reuse the layer ?
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # pool for rgb encoder
        self.d_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # pool for depth encoder
        self.un_pool = nn.MaxUnpool2d(kernel_size=2, stride=2)  # unpool for decoder

        # dropout layers
        self.drop = nn.Dropout(drop_out)

    def forward(self, x, dep):
        outputs = []
        # depth encoder
        # stage 1
        d0 = self.dlayer0(dep)
        d = self.d_pool(d0)

        # stage 2
        d1 = self.dlayer1(d)
        d = self.d_pool(d1)

        # stage 3
        d2 = self.dlayer2(d)
        d = self.d_pool(d2)
        d = self.drop(d)

        # stage 4
        d3 = self.dlayer3(d)
        d = self.d_pool(d3)
        d = self.drop(d)

        # stage 5
        d4 = self.dlayer4(d)

        # RGB encoder
        x = self.layer0(x)
        x = torch.add(x, d0)
        x = torch.div(x, 2)
        x, idx0 = self.pool(x)

        x = self.layer1(x)
        x = torch.add(x, d1)
        x = torch.div(x, 2)
        x, idx1 = self.pool(x)

        x = self.layer2(x)
        x = torch.add(x, d2)
        x = torch.div(x, 2)
        x, idx2 = self.pool(x)
        x = self.drop(x)

        x = self.layer3(x)
        x = torch.add(x, d3)
        x = torch.div(x, 2)
        x, idx3 = self.pool(x)
        x = self.drop(x)

        x = self.layer4(x)
        x = torch.add(x, d4)
        x = torch.div(x, 2)
        x_size = x.size()
        x, idx4 = self.pool(x)
        x = self.drop(x)

        # decoder
        y = self.un_pool(x, idx4, output_size=x_size)
        y = self.dec4(y)
        y = self.drop(y)

        y = self.un_pool(y, idx3)
        y = self.dec3(y)
        y = self.drop(y)

        y = self.un_pool(y, idx2)
        y = self.dec2(y)
        y = self.drop(y)

        y = self.un_pool(y, idx1)
        y = self.dec1(y)

        y = self.un_pool(y, idx0)
        y = self.dec0(y)

        outputs.append(y)
        return tuple(outputs)


def make_layers(cfg, in_channels, out_channels=None, batch_norm=False):
    layers = []
    for i, out_ch in enumerate(cfg):
        if i == len(cfg) - 1 and out_channels:
            out_ch = out_channels
        conv2d = nn.Conv2d(in_channels, out_ch, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = out_ch

    return nn.Sequential(*layers)


def get_fuse(dataset='nyud', pretrained=False, root='./encoding/models/pretrain', **kwargs):
    """pretrained: controls the layers other than the backbone"""
    # infer number of classes
    from ...datasets import datasets
    # backbone is already pretrained
    model = FuseNet(datasets[dataset.lower()].NUM_CLASS, layers, drop_out=0.5, **kwargs)
    print('layers after backbone pretrained {}'.format(pretrained))
    return model
