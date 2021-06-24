import torch
import torch.nn as nn
from encoding.models.model_store import get_model_file
import torchvision

layers = [[64, 64], [ 128, 128], [ 256, 256, 256], [ 512, 512, 512], [ 512, 512, 512]]


class FuseNet(nn.Module):

    def __init__(self, nclass, layers, init_weights=True, drop_out=0.5):
        super(FuseNet, self).__init__()
        # rgb encoder
        self.layer0 = make_layers(layers[0], 3, 64)
        self.layer1 = make_layers(layers[1], 64, 128)
        self.layer2 = make_layers(layers[2], 128, 256)
        self.layer3 = make_layers(layers[3], 256, 512)
        self.layer4 = make_layers(layers[4], 512, 512)

        # depth encoder: input 1 channel
        self.dlayer0 = make_layers(layers[0], 1, 64)
        self.dlayer1 = make_layers(layers[1], 64, 128)
        self.dlayer2 = make_layers(layers[2], 128, 256)
        self.dlayer3 = make_layers(layers[3], 256, 512)
        self.dlayer4 = make_layers(layers[4], 512, 512)

        # decoder layers
        self.dec4 = make_layers(layers[4], 512, 512)   # F:[512,512],[512,512],[512,512]
        self.dec3 = make_layers(layers[3], 512, 256)   # F:[512,512],[512,512],[512,256]
        self.dec2 = make_layers(layers[2], 256, 128)   # F:[256,256],[256,256],[256,128]
        self.dec1 = make_layers(layers[1], 128, 64 )   # F:[128,128],[128,64]
        self.dec0 = make_layers(layers[0], 64, nclass) # F:[64,64], [64,nclass]

        # pool layer has no params, so we can reuse the layer ?
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)    # pool for rgb encoder
        self.d_pool = nn.MaxPool2d(kernel_size=2, stride=2)                       # pool for depth encoder
        self.un_pool = nn.MaxUnpool2d(kernel_size=2, stride=2)                    # unpool for decoder

        # dropout layers
        self.drop = nn.Dropout(drop_out)

    def forward(self, x, dep):

        outputs = []
        # depth encoder
        d0 = self.dlayer0(dep)
        d1 = self.dlayer1(self.d_pool(d0))
        d2 = self.dlayer2(self.d_pool(d1))
        d3 = self.dlayer3(self.drop(self.d_pool(d2)))
        d4 = self.dlayer4(self.drop(self.d_pool(d3)))

        # RGB encoder
        x0, idx0 = self.pool(self.layer0(x) + d0)
        x1, idx1 = self.pool(self.layer1(x0) + d1)
        x2, idx2 = self.pool(self.layer2(x1) + d2)
        x2 = self.drop(x2)
        x3, idx3 = self.pool(self.layer3(x2) + d3)
        x3 = self.drop(x3)
        x4, idx4 = self.pool(self.layer4(x3) + d4)
        x4 = self.drop(x4)

        # decoder
        y3 = self.dec4(self.un_pool(x4, idx4))
        y3 = self.drop(y3)
        y2 = self.dec3(self.un_pool(y3, idx3))
        y2 = self.drop(y2)
        y1 = self.dec2(self.un_pool(y2, idx2))
        y1 = self.drop(y1)
        y0 = self.dec1(self.un_pool(y1, idx1))
        y = self.dec0(self.un_pool(y0, idx0))

        outputs.append(y)
        return tuple(outputs)

def make_layers(cfg, in_channels, out_channels=None, batch_norm=False):
    layers = []
    for i, out_ch in enumerate(cfg):
        if i==len(cfg)-1 and out_channels:
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
    from ...datasets import datasets, acronyms
    # backbone is already pretrained
    model = FuseNet(datasets[dataset.lower()].NUM_CLASS, layers, drop_out = 0.5, **kwargs)
    print('layers after backbone pretrained {}'.format(pretrained))
    return model