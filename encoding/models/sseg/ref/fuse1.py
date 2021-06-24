import torch
import torch.nn as nn
from encoding.models.model_store import get_model_file


__all__ = [
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

class FuseNet(nn.Module):

    def __init__(self, nclass, layers, init_weights=True, drop_out=None):
        super(FuseNet, self).__init__()
        # two encoder branch: input for depth-encoder: 1 channel
        self.layer0 = make_layers(layers[0], 3)
        self.layer1 = make_layers(layers[1], layers[0][-1])
        self.layer2 = make_layers(layers[2], layers[1][-1])
        self.layer3 = make_layers(layers[3], layers[2][-1])
        self.layer4 = make_layers(layers[4], layers[3][-1])

        self.dlayer0 = make_layers(layers[0], 1)
        self.dlayer0 = make_layers(layers[1], layers[0][-1])
        self.dlayer2 = make_layers(layers[2], layers[1][-1])
        self.dlayer3 = make_layers(layers[3], layers[2][-1])
        self.dlayer4 = make_layers(layers[4], layers[3][-1])

        self.dec4 = make_layers(layers[4], layers[4][-1])  # last encoder block is layers[4]
        self.dec3 = make_layers(layers[3], layers[4][-1])
        self.dec2 = make_layers(layers[2], layers[3][-1])
        self.dec1 = make_layers(layers[1], layers[2][-1])
        self.dec0 = make_layers(layers[0], layers[1][-1], out_chan=nclass)

        # pool layer has no params, so we can reuse the layer ?
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.d_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # dropout layers
        if drop_out:
            self.layer2_drop = nn.Dropout(drop_out)
            self.layer3_drop = nn.Dropout(drop_out)
            self.layer4_drop = nn.Dropout(drop_out)
            self.dlayer2_drop = nn.Dropout(drop_out)
            self.dlayer3_drop = nn.Dropout(drop_out)
            self.dec4_drop = nn.Dropout(drop_out)
            self.dec3_drop = nn.Dropout(drop_out)
            self.dec2_drop = nn.Dropout(drop_out)

        if init_weights:
            self._initialize_weights()

    def forward(self, x, dep):

        outputs = []
        # depth encoder
        d0 = self.dlayer0(dep)
        d1 = self.dlayer1(self.pool(d0))
        d2 = self.dlayer2(self.pool(d1))
        d3 = self.dlayer3(self.pool(d2))
        d4 = self.dlayer4(self.pool(d3))

        # RGB encoder
        x0, idx0 = self.pool(self.layer0(x) + d0)
        x1, idx1 = self.pool(self.layer1(x0) + d1)
        x2, idx2 = self.pool(self.layer2(x1) + d2)
        x2 = self.layer2_drop(x2)
        x3, idx3 = self.pool(self.layer3(x2) + d3)
        x3 = self.layer3_drop(x3)
        x4, idx4 = self.pool(self.layer4(x3) + d4)
        x4 = self.layer5_drop(x4)

        # decoder
        y3 = self.dec4(self.unpool(x4, idx4))
        y3 = self.dec4_drop(y3)
        y2 = self.dec3(self.unpool(y3, idx3))
        y2 = self.dec3_drop(y2)
        y1 = self.dec2(self.unpool(y2, idx2))
        y1 = self.dec2_drop(y1)
        y0 = self.dec1(self.unpool(y1, idx1))
        y = self.dec1(self.unpool(y0, idx0))

        outputs.append(y)
        return tuple(outputs)

    def _initialize_weights(self, root='./encoding/models/pretrain'):
        pretrain_param = torch.load(get_model_file('vgg16', root=root))




def make_layers(cfg, in_chan=None, out_chan=None, batch_norm=False):
    layers = []
    in_channels = in_chan if in_chan else cfg[0]
    for i, v in enumerate(cfg):
        if out_chan and i==len(cfg)-1:
            v = out_chan
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v

    return nn.Sequential(*layers)


cfgs = {
    #'A': [[64, 'M'], [128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    #'B': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 'M'], [512, 512, 'M'], [512, 512, 'M']],
    'D': [[64, 64], [ 128, 128], [ 256, 256, 256], [ 512, 512, 512], [ 512, 512, 512], ],
    #'E': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 256, 'M'], [512, 512, 512, 512, 'M'], [512, 512, 512, 512, 'M']],
}
pretrain_param = torch.load(get_model_file('vgg16', root=root))
model = FuseNet(40, cfgs['D'], init_weights=False)

my_kv=model.state_dict()
l1_kv = model.layer0.parameters().state_dict()

# my fake code
for p in model.parameters():
    if p.requires_grad:
         print(p.name)

def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
