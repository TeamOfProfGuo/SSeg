
import torch
import torch.nn as nn
from torch.nn import functional as F

from math import log

__all__ = ['ResidualConvUnit', 'SE_Block', 'MultiResolutionFusion',
            'MRF_Concat_5_2', 'BaseRefineNetBlock', 'RGBD_Fuse_Block']

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

class RGBD_Fuse_Block(nn.Module):
    def __init__(self, in_feats, out_method='concat', pre_module='se', mode='late', pre_setting={}): # out_feats=None, 
        super().__init__()
        module_dict = {'gc': GC_Block, 'se': SE_Block, 'spa': SPA_Block, 'pp': PP_Block, 'eca': ECA_Block, 'scse': SCSE_Block}
        self.mode = mode
        self.module = module_dict[pre_module]
        self.out_method = out_method
        if mode == 'late':
            self.rgb_pre = self.module(in_feats, **pre_setting)
            self.dep_pre = self.module(in_feats, **pre_setting)
        elif mode == 'early':
            self.pre = self.module(2 * in_feats, **pre_setting)
        else:
            raise ValueError('Invalid Fuse Mode: ' + str(mode))
        if out_method == 'concat':
            self.out_block = nn.Sequential(
                nn.Conv2d(2 * in_feats, in_feats, kernel_size=1, stride=1, bias=False)
            )

    def forward(self, rgb, dep):
        if self.mode == 'late':
            rgb = self.rgb_pre(rgb)
            dep = self.dep_pre(dep)
            return (rgb + dep) if self.out_method == 'add' else (self.out_block(torch.cat((rgb, dep), 1)))
        elif self.mode == 'early':
            _, c, _, _ = rgb.size()
            feats = torch.cat((rgb, dep), dim=1)
            feats = self.pre(feats)
            if self.out_method == 'add':
                rgb = feats[:, :c, :, :]
                dep = feats[:, c:, :, :]
                return rgb + dep
            else:
                return self.out_block(feats)
        else:
            raise ValueError('Invalid Fuse Mode: ' + str(self.mode))

class GC_Block(nn.Module):
    def __init__(self, in_feats, mid_factor=4):
        super().__init__()
        mid_feats = in_feats // mid_factor
        self.softmax = nn.Softmax(dim=-1)
        self.att_conv = nn.Conv2d(in_feats, 1, kernel_size=1)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(in_feats, mid_feats, kernel_size=1),
            nn.LayerNorm([mid_feats, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_feats, in_feats, kernel_size=1)
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        x_in = x.view(b, 1, c, h*w)                                   # [b, 1, c, h*w]
        context_mask = self.att_conv(x).view(b, 1, -1)                # [b, 1, h*w]
        context_mask = self.softmax(context_mask).unsqueeze(-1)       # [b, 1, h*w, 1]
        context = torch.matmul(x_in, context_mask).view(b, -1, 1, 1)  # [b, c, 1, 1]
        out = x * self.channel_add_conv(context)
        return out

class SE_Block(nn.Module):
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

class SPA_Block(nn.Module):
    def __init__(self, in_feats, reduction=16):
        super().__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        pool_feats_num = 21 * in_feats
        self.fc = nn.Sequential(
            nn.Linear(pool_feats_num, pool_feats_num // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(pool_feats_num // reduction, in_feats, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x).view(b, c)  # like resize() in numpy
        y2 = self.avg_pool2(x).view(b, 4 * c)
        y3 = self.avg_pool4(x).view(b, 16 * c)
        y = torch.cat((y1, y2, y3), 1)
        weighting = self.fc(y)
        b, out_channel = weighting.size()
        weighting = weighting.view(b, out_channel, 1, 1)
        # return weighting
        out = weighting * x
        return out

class PP_Block(nn.Module):
    def __init__(self, in_feats, pp_layer=3, reduction=16):
        super().__init__()
        self.pp_layer = pp_layer
        self.pp_size = 2 ** (pp_layer-1)
        self.fc = nn.Sequential(
            nn.Conv2d(pp_layer * in_feats, in_feats, kernel_size=1),
            nn.BatchNorm2d(in_feats),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_feats, in_feats // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_feats // reduction, in_feats, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # b, c, _, _ = x.size()
        pooling_pyramid = []
        for i in range(self.pp_layer):
            pooling_pyramid.append(F.interpolate(F.adaptive_avg_pool2d(x, 2 ** i), size=self.pp_size))
        y = torch.cat(tuple(pooling_pyramid), dim=1)
        weighting = self.fc(y)
        return weighting * x

class ECA_Block(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, in_feats, gamma=2, b=1):
        super(ECA_Block, self).__init__()
        t = int(abs((log(in_feats, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class SCSE_Block(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch, ch // re,1),
                                 nn.ReLU(inplace = True),
                                 nn.Conv2d(ch // re, ch, 1),
                                 nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(ch, ch, 1),
                                 nn.Sigmoid())
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

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
    def __init__(self, features, residual_conv_unit, multi_resolution_fusion, *shapes):
        super().__init__()

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

        return self.output_conv(out)

# class MultiResolutionFusion_Concat(nn.Module):
#     def __init__(self, out_feats, *shapes):
#         super().__init__()
#         _, min_scale = min(shapes, key=lambda x: x[1])

#         self.scale_factors = []
#         for i, shape in enumerate(shapes):
#             feat, scale= shape
#             self.scale_factors.append(scale // min_scale)
#             self.add_module("resolve{}".format(i),
#                             nn.Conv2d(feat, out_feats, kernel_size=3, stride=1, padding=1, bias=False))
        
#         self.out_block = nn.Sequential(
#             nn.Conv2d(len(shapes)*out_feats, out_feats, kernel_size=1, stride=1, padding=0, bias=False),
#             # nn.BatchNorm2d(out_feats), nn.ReLU(inplace=True),
#             # nn.Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=False)
#         )

#     def forward(self, *xs):

#         concat_feat = [self.resolve0(xs[0])]
#         if self.scale_factors[0] != 1:
#             concat_feat[0] = nn.functional.interpolate(concat_feat[0], scale_factor=self.scale_factors[0], mode='bilinear', align_corners=True)

#         for i, x in enumerate(xs[1:], 1): # the value for i starts from 1
#             out = self.__getattr__("resolve{}".format(i))(x)
#             if self.scale_factors[i] != 1:
#                 out = nn.functional.interpolate(out, scale_factor=self.scale_factors[i], mode='bilinear', align_corners=True)
#             concat_feat.append(out)
        
#         output = torch.cat(tuple(concat_feat), 1)
#         output = self.out_block(output)
#         return output


# class ChainedResidualPool(nn.Module):
#     def __init__(self, feats):
#         super().__init__()

#         self.relu = nn.ReLU(inplace=True)
#         for i in range(1, 4):
#             self.add_module("block{}".format(i),
#                             nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
#                                           nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False))
#                             )

#     def forward(self, x):
#         x = self.relu(x)
#         path = x

#         for i in range(1, 4):
#             path = self.__getattr__("block{}".format(i))(path)
#             x = x + path

#         return x


# class ChainedResidualPoolImproved(nn.Module):
#     def __init__(self, feats):
#         super().__init__()

#         self.relu = nn.ReLU(inplace=True)
#         for i in range(1, 5):
#             self.add_module("block{}".format(i),
#                             nn.Sequential(nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False),
#                                           nn.MaxPool2d(kernel_size=5, stride=1, padding=2)))

#     def forward(self, x):
#         x = self.relu(x)
#         path = x

#         for i in range(1, 5):
#             path = self.__getattr__("block{}".format(i))(path)
#             x += path

#         return x