import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

__all__ = ['RN_gap_fuse', 'get_rn_gap_fuse']

class RN_gap_fuse(nn.Module):

    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, root='./encoding/models/pretrain',
                 module_setting={}):
        super(RN_gap_fuse, self).__init__()
        # self.do = nn.Dropout(p=0.5)

        self.early_fusion = module_setting.ef if module_setting.ef is not None else False
        self.use_crp = module_setting.crp if module_setting.crp is not None else False
        n_features = module_setting.n_features if module_setting.n_features is not None else 256

        self.rgb_base = get_resnet18(input_dim=4) if self.early_fusion else get_resnet18(input_dim=3)
        self.dep_base = get_resnet18(input_dim=1)
                
        self.rgb_layer0 = nn.Sequential(self.rgb_base.conv1, self.rgb_base.bn1, 
                                        self.rgb_base.relu, self.rgb_base.maxpool)  # [B, 64, h/4, w/4]
        self.rgb_layer1 = self.rgb_base.layer1  # [B, 64, h/4, w/4]
        self.rgb_layer2 = self.rgb_base.layer2  # [B, 128, h/8, w/8]
        self.rgb_layer3 = self.rgb_base.layer3  # [B, 256, h/16, w/16]
        self.rgb_layer4 = self.rgb_base.layer4  # [B, 512, h/32, w/32]

        self.dep_layer0 = nn.Sequential(self.dep_base.conv1, self.dep_base.bn1,
                                        self.dep_base.relu, self.dep_base.maxpool)  # [B, 64, h/4, w/4]
        self.dep_layer1 = self.dep_base.layer1  # [B, 64, h/4, w/4]
        self.dep_layer2 = self.dep_base.layer2  # [B, 128, h/8, w/8]
        self.dep_layer3 = self.dep_base.layer3  # [B, 256, h/16, w/16]
        self.dep_layer4 = self.dep_base.layer4  # [B, 512, h/32, w/32]

        self.fuse1 = GAP_Fuse_Block( 64, n_features)
        self.fuse2 = GAP_Fuse_Block(128, n_features)
        self.fuse3 = GAP_Fuse_Block(256, n_features)
        self.fuse4 = GAP_Fuse_Block(512, 2*n_features)

        self.refine4 = RefineNetBlock(2*n_features, self.use_crp, (2*n_features, 32))
        self.refine3 = RefineNetBlock(n_features, self.use_crp, (2*n_features, 32), (n_features, 16))
        self.refine2 = RefineNetBlock(n_features, self.use_crp, (n_features, 16), (n_features, 8))
        self.refine1 = RefineNetBlock(n_features, self.use_crp, (n_features, 8), (n_features, 4))

        self.out_conv = nn.Sequential(
            ResidualConvUnit(n_features), ResidualConvUnit(n_features),
            nn.Conv2d(n_features, n_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x, d):
        _, _, h, w = x.size()

        x = self.rgb_layer0(x)     # [B, 64, h/4, w/4]
        l1 = self.rgb_layer1(x)    # [B, 64, h/4, w/4]
        l2 = self.rgb_layer2(l1)   # [B, 128, h/8, w/8]
        l3 = self.rgb_layer3(l2)   # [B, 256, h/16, w/16]
        l4 = self.rgb_layer4(l3)   # [B, 512, h/32, w/32]

        d = self.dep_layer0(d)     # [B, 64, h/4, w/4]
        d1 = self.dep_layer1(d)    # [B, 64, h/4, w/4]
        d2 = self.dep_layer2(d1)   # [B, 128, h/8, w/8]
        d3 = self.dep_layer3(d2)   # [B, 256, h/16, w/16]
        d4 = self.dep_layer4(d3)   # [B, 512, h/32, w/32]

        f1 = self.fuse1(l1, d1)
        f2 = self.fuse2(l2, d2)
        f3 = self.fuse3(l3, d3)
        f4 = self.fuse4(l4, d4)

        path4 = self.refine4(f4)          # [B, 512, h/32, w/32]
        path3 = self.refine3(path4, f3)   # [B, 256, h/16, w/16]
        path2 = self.refine2(path3, f2)   # [B, 256, h/8, w/8]
        path1 = self.refine1(path2, f1)   # [B, 256, h/4, w/4]

        out = self.out_conv(path1)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out

def get_rn_gap_fuse(dataset='nyud', backbone='resnet18', pretrained=True, root='./encoding/models/pretrain/', module_setting={}):
    from ...datasets import datasets
    model = RN_gap_fuse(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, pretrained=pretrained, root=root, module_setting=module_setting)
    return model

def get_resnet18(pretrained=True, input_dim = 3, f_path='./encoding/models/pretrain/resnet18-5c106cde.pth'):
    assert input_dim in (1, 3, 4)
    model = models.resnet18(pretrained=False)

    if pretrained:
        # Check weights file
        if not os.path.exists(f_path):
            raise FileNotFoundError('The pretrained model cannot be found.')
        
        if input_dim != 3:
            model.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            weights = torch.load(f_path)
            for k, v in weights.items():
                weights[k] = v.data
            conv1_ori = weights['conv1.weight']
            conv1_new = torch.zeros((64, input_dim, 7, 7), dtype=torch.float32)
            if input_dim == 4:
                conv1_new[:, :3, :, :] = conv1_ori
                conv1_new[:,  3, :, :] = conv1_ori[:,  1, :, :]
            else:
                conv1_new[:,  0, :, :] = conv1_ori[:,  1, :, :]
            weights['conv1.weight'] = conv1_new
            model.load_state_dict(weights, strict=False)
        else:
            model.load_state_dict(torch.load(f_path), strict=False)
    else:
        raise ValueError('Please use pretrained resnet18.')
    
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

class GAP_Fuse_Block(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.rgb_weighting = GAP_Weighting_Block(in_feats)
        self.dep_weighting = GAP_Weighting_Block(in_feats)
        self.out_conv = nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, rgb, dep):
        reweighted_rgb = self.rgb_weighting(rgb)
        reweighted_dep = self.dep_weighting(dep)
        out = reweighted_rgb + reweighted_dep
        return self.out_conv(out)

class GAP_Weighting_Block(nn.Module):
    def __init__(self, in_feats, mid_factor=4):
        super().__init__()
        self.mid_feats = in_feats // mid_factor
        self.fc = nn.Sequential(
            nn.Conv2d(in_feats, self.mid_feats, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_feats, in_feats, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = weighting * x
        return y

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

class MultiResolutionFusion_Concat(nn.Module):
    def __init__(self, out_feats, *shapes):
        super().__init__()
        _, min_scale = min(shapes, key=lambda x: x[1])

        self.scale_factors = []
        for i, shape in enumerate(shapes):
            feat, scale= shape
            self.scale_factors.append(scale // min_scale)
            self.add_module("resolve{}".format(i),
                            nn.Conv2d(feat, out_feats, kernel_size=3, stride=1, padding=1, bias=False))
        
        self.out_block = nn.Sequential(
            nn.Conv2d(len(shapes)*out_feats, out_feats, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(out_feats), nn.ReLU(inplace=True),
            # nn.Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, *xs):

        concat_feat = [self.resolve0(xs[0])]
        if self.scale_factors[0] != 1:
            concat_feat[0] = nn.functional.interpolate(concat_feat[0], scale_factor=self.scale_factors[0], mode='bilinear', align_corners=True)

        for i, x in enumerate(xs[1:], 1): # the value for i starts from 1
            out = self.__getattr__("resolve{}".format(i))(x)
            if self.scale_factors[i] != 1:
                out = nn.functional.interpolate(out, scale_factor=self.scale_factors[i], mode='bilinear', align_corners=True)
            concat_feat.append(out)
        
        output = torch.cat(tuple(concat_feat), 1)
        output = self.out_block(output)
        return output

class MRF_Concat_5_2(nn.Module):
    def __init__(self, out_feats, *shapes):
        super().__init__()
        _, min_scale = min(shapes, key=lambda x: x[1])

        self.scale_factors = []
        self.concat_count = 0
        for shape in shapes:
            self.concat_count += shape[0]
            self.scale_factors.append(shape[1] // min_scale)
        
        self.out_block = nn.Sequential(
            nn.BatchNorm2d(self.concat_count), nn.ReLU(inplace=True),
            nn.Conv2d(self.concat_count, out_feats, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, *xs):

        concat_feat = [xs[0]]
        if self.scale_factors[0] != 1:
            concat_feat[0] = nn.functional.interpolate(concat_feat[0], scale_factor=self.scale_factors[0], mode='bilinear', align_corners=True)

        for i, x in enumerate(xs[1:], 1): # the value for i starts from 1
            if self.scale_factors[i] != 1:
                x = nn.functional.interpolate(x, scale_factor=self.scale_factors[i], mode='bilinear', align_corners=True)
            concat_feat.append(x)
        
        output = torch.cat(tuple(concat_feat), 1)
        output = self.out_block(output)
        return output

class MRF_Concat_6_2(nn.Module):
    def __init__(self, out_feats, *shapes):
        super().__init__()
        _, min_scale = min(shapes, key=lambda x: x[1])

        self.scale_factors = []
        self.concat_count = 0
        for shape in shapes:
            self.concat_count += shape[0]
            self.scale_factors.append(shape[1] // min_scale)
        
        self.out_block = nn.Conv2d(self.concat_count, out_feats, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, *xs):

        concat_feat = [xs[0]]
        if self.scale_factors[0] != 1:
            concat_feat[0] = nn.functional.interpolate(concat_feat[0], scale_factor=self.scale_factors[0], mode='bilinear', align_corners=True)

        for i, x in enumerate(xs[1:], 1): # the value for i starts from 1
            if self.scale_factors[i] != 1:
                x = nn.functional.interpolate(x, scale_factor=self.scale_factors[i], mode='bilinear', align_corners=True)
            concat_feat.append(x)
        
        output = torch.cat(tuple(concat_feat), 1)
        output = self.out_block(output)
        return output

class BaseRefineNetBlock(nn.Module):
    def __init__(self, features, use_crp, residual_conv_unit, multi_resolution_fusion, chained_residual_pool, *shapes):
        super().__init__()

        for i, shape in enumerate(shapes):
            feats = shape[0]
            self.add_module("rcu{}".format(i), nn.Sequential(residual_conv_unit(feats), residual_conv_unit(feats)))

        self.mrf = multi_resolution_fusion(features, *shapes) if len(shapes) != 1 else None
        self.crp = chained_residual_pool(features) if use_crp else None
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        rcu_xs = []

        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__("rcu{}".format(i))(x))

        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]

        if self.crp is not None:
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


# Choices of MultiResolutionFusion Block
MRF_DICT = {'RefineNet': MultiResolutionFusion, 'Concat_5_2': MRF_Concat_5_2, 'Concat_6_2': MRF_Concat_6_2}


class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, use_crp, *shapes):
        super().__init__(features, use_crp, ResidualConvUnit, MRF_DICT['Concat_6_2'], ChainedResidualPool, *shapes)