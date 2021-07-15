import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

__all__ = ['RefineNet', 'get_refinenet']

class RefineNet(nn.Module):

    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, root='./encoding/models/pretrain',
                 module_setting={}):
        super(RefineNet, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.base = models.resnet18(pretrained=False)

        self.early_fusion = module_setting['ef'] if module_setting['ef'] is not None else False
        self.use_crp = module_setting['crp'] if module_setting['crp'] is not None else False
        n_features = module_setting['n_features'] if module_setting['n_features'] is not None else 256

        if pretrained:
            print('[root]:', root)
            if backbone == 'resnet18':
                # f_path = os.path.abspath(os.path.join(root, 'resnet18-5c106cde.pth'))
                f_path = '/gpfsnyu/scratch/hl3797/DANet/encoding/models/pretrain/resnet18-5c106cde.pth'
                print('[Pretrain Weights]:', f_path)
            if not os.path.exists(f_path):
                raise FileNotFoundError('the pretrained model can not be found')
            # Set up weights for early fusion
            if self.early_fusion:
                self.base.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
                weights = torch.load(f_path)
                for k, v in weights.items():
                    weights[k] = v.data
                conv1_3c = weights['conv1.weight']
                conv1_4c = torch.zeros((64, 4, 7, 7), dtype=torch.float32)
                conv1_4c[:, :3, :, :] = conv1_3c
                conv1_4c[:,  3, :, :] = conv1_3c[:,  1, :, :]
                weights['conv1.weight'] = conv1_4c
                self.base.load_state_dict(weights, strict=False)
            else:
                self.base.load_state_dict(torch.load(f_path), strict=False)
                
        self.in_block = nn.Sequential(self.base.conv1, self.base.bn1, self.base.relu, self.base.maxpool)  # [B, 64, h/4, w/4]

        self.layer1 = self.base.layer1  # [B, 64, h/4, w/4]
        self.layer2 = self.base.layer2  # [B, 128, h/8, w/8]
        self.layer3 = self.base.layer3  # [B, 256, h/16, w/16]
        self.layer4 = self.base.layer4  # [B, 512, h/32, w/32]

        self.layer4_rn = nn.Conv2d(512, 2*n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(256, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(128, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_rn = nn.Conv2d(64,  n_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.refine4 = RefineNetBlock(2*n_features, self.use_crp, (2*n_features, 32))
        self.refine3 = RefineNetBlock(n_features, self.use_crp, (2*n_features, 32), (n_features, 16))
        self.refine2 = RefineNetBlock(n_features, self.use_crp, (n_features, 16), (n_features, 8))
        self.refine1 = RefineNetBlock(n_features, self.use_crp, (n_features, 8), (n_features, 4))

        self.out_conv = nn.Sequential(
            ResidualConvUnit(n_features), ResidualConvUnit(n_features),
            nn.Conv2d(n_features, n_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.in_block(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)  # [B, 256, h/16, w/16]
        l4 = self.layer4(l3)  # [B, 512, h/32, w/32]

        # l4 = self.do(l4)
        # l3 = self.do(l3)
        l1 = self.layer1_rn(l1)
        l2 = self.layer2_rn(l2)
        l3 = self.layer3_rn(l3)
        l4 = self.layer4_rn(l4)  # [B, 512, h/32, w/32]

        path4 = self.refine4(l4)          # [B, 512, h/32, w/32]
        path3 = self.refine3(path4, l3)   # [B, 256, h/16, w/16]
        path2 = self.refine2(path3, l2)   # [B, 256, h/8, w/8]
        path1 = self.refine1(path2, l1)

        out = self.out_conv(path1)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out


def get_refinenet(dataset='nyud', backbone='resnet18', pretrained=True, root='./encoding/models/pretrain', module_setting={}):
    from ...datasets import datasets
    model = RefineNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, pretrained=pretrained, root=root, module_setting=module_setting)
    return model


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class MultiResolutionFusion(nn.Module):
    def __init__(self, out_feats, *shapes):
        super().__init__()
        _, min_scale = min(shapes, key=lambda x: x[1])

        self.scale_factors = []
        for i, shape in enumerate(shapes):
            feat, scale= shape
            self.scale_factors.append(scale // min_scale)
            self.add_module("resolve{}".format(i),
                            nn.Conv2d(feat, out_feats, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, *xs):

        output = self.resolve0(xs[0])
        if self.scale_factors[0] != 1:
            output = nn.functional.interpolate(output, scale_factor=self.scale_factors[0], mode='bilinear', align_corners=True)

        for i, x in enumerate(xs[1:], 1): # the value for i starts from 1
            current_out = self.__getattr__("resolve{}".format(i))(x)
            if self.scale_factors[i] != 1:
                current_out = nn.functional.interpolate(current_out, scale_factor=self.scale_factors[i], mode='bilinear', align_corners=True)
            output += current_out

        return output


class BaseRefineNetBlock(nn.Module):
    def __init__(self, features, use_crp, residual_conv_unit, multi_resolution_fusion, chained_residual_pool, *shapes):
        super().__init__()

        self.use_crp = use_crp
        if use_crp:
            self.crp = chained_residual_pool(features)

        for i, shape in enumerate(shapes):
            feats = shape[0]
            self.add_module("rcu{}".format(i), nn.Sequential(residual_conv_unit(feats), residual_conv_unit(feats)))

        self.mrf = multi_resolution_fusion(features, *shapes) if len(shapes) != 1 else None
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        rcu_xs = []

        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__("rcu{}".format(i))(x))

        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]

        if self.use_crp:
            out = self.crp(out)

        return self.output_conv(out)


class ChainedResidualPool(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 4):
            self.add_module("block{}".format(i),
                            nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                                          nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False))
                            )

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 4):
            path = self.__getattr__("block{}".format(i))(path)
            x = x + path

        return x


class ChainedResidualPoolImproved(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 5):
            self.add_module("block{}".format(i),
                            nn.Sequential(nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.MaxPool2d(kernel_size=5, stride=1, padding=2)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 5):
            path = self.__getattr__("block{}".format(i))(path)
            x += path

        return x


class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, use_crp, *shapes):
        super().__init__(features, use_crp, ResidualConvUnit, MultiResolutionFusion, ChainedResidualPool, *shapes)
