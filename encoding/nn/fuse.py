
import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ['ResidualConvUnit', 'GAP_Fuse_Block', 'SE_Block', 'MultiResolutionFusion',
            'MRF_Concat_5_2', 'MRF_Concat_6_2', 'BaseRefineNetBlock', 'RGBD_Fuse_Block']

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
    def __init__(self, in_feats, out_method='concat', pre_module='context', pre_setting={}): # out_feats=None, 
        super().__init__()
        module_dict = {'context': Global_Context_Block, 'se': SE_Block}
        self.module = module_dict[pre_module]
        self.rgb_pre = self.module(in_feats, **pre_setting)
        self.dep_pre = self.module(in_feats, **pre_setting)

        self.out_method = out_method
        if out_method == 'concat':
            self.out_block = nn.Sequential(
                nn.Conv2d(2 * in_feats, in_feats, kernel_size=1, stride=1, bias=False)
            )

    def forward(self, rgb, dep):
        rgb = self.rgb_pre(rgb)
        dep = self.dep_pre(dep)
        return (rgb + dep) if self.out_method == 'add' else (self.out_block(torch.cat((rgb, dep), 1)))

class Global_Context_Block(nn.Module):
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
        out = x + self.channel_add_conv(context)
        return out


class GAP_Fuse_Block(nn.Module):
    def __init__(self, in_feats, out_feats, out_conv=True):
        super().__init__()
        self.rgb_weighting = SE_Block(in_feats)
        self.dep_weighting = SE_Block(in_feats)
        self.flag = out_conv
        self.out_conv = nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, rgb, dep):
        reweighted_rgb = self.rgb_weighting(rgb)
        reweighted_dep = self.dep_weighting(dep)
        out = reweighted_rgb + reweighted_dep
        return self.out_conv(out) if self.flag else out

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