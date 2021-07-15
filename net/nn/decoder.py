
import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ['Decoder']

class Decoder(nn.Module):
    def __init__(self, n_features, n_classes, decoder='base'):
        super().__init__()
        decoder_dict = {'base': Base_Decoder, 'refine': Refine_Decoder, 'pa': PA_Decoder}
        self.decoder = decoder_dict[decoder](n_features, n_classes)
    
    def forward(self, feats):
        return self.decoder(feats)

class Base_Decoder(nn.Module):
    def __init__(self, n_features, n_classes, conv_module='cbr', level_fuse='simple', out='rgb'):
        super().__init__()

        self.out = out

        level_fuse_dict = {'simple': Simple_Level_Fuse, 'gau': GAU_Block}
        self.refine2 = level_fuse_dict[level_fuse](64)
        self.refine3 = level_fuse_dict[level_fuse](128)
        self.refine4 = level_fuse_dict[level_fuse](256)

        if conv_module == 'rcu':
            self.up2 = nn.Sequential(ResidualConvUnit(128), LearnedUpUnit(128, 64))
            self.up3 = nn.Sequential(ResidualConvUnit(256), LearnedUpUnit(256, 128))
            self.up4 = nn.Sequential(ResidualConvUnit(512), LearnedUpUnit(512, 256))
            self.out_conv = nn.Sequential(
                ResidualConvUnit(64), ResidualConvUnit(64),
                nn.Conv2d(n_features, n_classes, kernel_size=1, stride=1, padding=0, bias=True))
        elif conv_module == 'cbr':
            self.up2 = nn.Sequential(CBR(128,  64), LearnedUpUnit(64))
            self.up3 = nn.Sequential(CBR(256, 128), LearnedUpUnit(128))
            self.up4 = nn.Sequential(CBR(512, 256), LearnedUpUnit(256))
            self.out_conv = nn.Sequential(
                CBR(64, n_features), CBR(n_features, n_features),
                nn.Conv2d(n_features, n_classes, kernel_size=1, stride=1, padding=0, bias=True))
        elif conv_module == 'bb':
            self.up2 = nn.Sequential(BasicBlock(128, 128), BasicBlock(128,  64, upsample=True))
            self.up3 = nn.Sequential(BasicBlock(256, 256), BasicBlock(256, 128, upsample=True))
            self.up4 = nn.Sequential(BasicBlock(512, 512), BasicBlock(512, 256, upsample=True))
            self.out_conv = nn.Sequential(
                BasicBlock(64, n_features, upsample=True), BasicBlock(n_features, n_features),
                nn.Conv2d(n_features, n_classes, kernel_size=1, stride=1, padding=0, bias=True))
        else:
            raise ValueError('Invalid conv module: %s.' % conv_module)

    def forward(self, feats):
        if self.out == 'rgb':
            l1, l2, l3, l4 = feats.l1, feats.l2, feats.l3, feats.l4
        elif self.out == 'dep':
            l1, l2, l3, l4 = feats.d1, feats.d2, feats.d3, feats.d4
        else:
            raise ValueError('Invalid out feats: %s.' % self.out)

        y4 = self.up4(l4)          # [B, 256, h/16, w/16]
        y3 = self.refine4(y4, l3)  # [B, 256, h/16, w/16]

        y3 = self.up3(y3)          # [B, 128, h/8, w/8]
        y2 = self.refine3(y3, l2)  # [B, 128, h/8, w/8]

        y2 = self.up2(y2)          # [B, 64, h/4, w/4]
        y1 = self.refine2(y2, l1)  # [B, 64, h/4, w/4]

        return self.out_conv(y1)

class PA_Decoder(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()

        self.refine4 = FPA(512)
        self.refine3 = GAU(512, 256)
        self.refine2 = GAU(256, 128)
        self.refine1 = GAU(128,  64)
        
        self.out_conv = nn.Sequential(
                CBR(64, n_features), CBR(n_features, n_features),
                nn.Conv2d(n_features, n_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, feats):
        l1, l2, l3, l4 = feats.l1, feats.l2, feats.l3, feats.l4
        feats = self.refine4(l4)         # [B, 512, h/32, w/32]
        feats = self.refine3(feats, l3)  # [B, 256, h/16, w/16]
        feats = self.refine2(feats, l2)  # [B, 128, h/8, w/8]
        feats = self.refine1(feats, l1)  # [B, 64, h/4, w/4]
        return self.out_conv(feats)

class Refine_Decoder(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()

        self.refine_conv1 = nn.Conv2d( 64, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.refine_conv2 = nn.Conv2d(128, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.refine_conv3 = nn.Conv2d(256, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.refine_conv4 = nn.Conv2d(512, 2*n_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.refine4 = RefineNetBlock(2*n_features, (2*n_features, 32))
        self.refine3 = RefineNetBlock(n_features, (2*n_features, 32), (n_features, 16))
        self.refine2 = RefineNetBlock(n_features, (n_features, 16), (n_features, 8))
        self.refine1 = RefineNetBlock(n_features, (n_features, 8), (n_features, 4))

        self.out_conv = nn.Sequential(
            ResidualConvUnit(n_features), ResidualConvUnit(n_features),
            nn.Conv2d(n_features, n_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, feats):
        l1 = self.refine_conv1(feats.l1)
        l2 = self.refine_conv2(feats.l2)
        l3 = self.refine_conv3(feats.l3)
        l4 = self.refine_conv4(feats.l4)

        y4 = self.refine4(l4)       # [B, 512, h/32, w/32]
        y3 = self.refine3(y4, l3)   # [B, 256, h/16, w/16]
        y2 = self.refine2(y3, l2)   # [B, 256, h/8, w/8]
        y1 = self.refine1(y2, l1)   # [B, 256, h/4, w/4]

        return self.out_conv(y1)

class FPA(nn.Module):
    def __init__(self, channels=512):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super().__init__()
        channels_mid = channels // 4

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channels)

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Convolution Upsample
        self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)

        # Merge branch 1 and 2
        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))
        x1_merge = self.relu(x1_2 + x2_upsample)

        x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))

        #
        out = self.relu(x_master + x_gpb)

        return out

class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        return out

class LearnedUpUnit(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.dep_conv = nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1, padding=1, groups=in_feats, bias=False)

    def forward(self, x):
        x = self.up(x)
        x = self.dep_conv(x)
        return x

class Simple_Level_Fuse(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        
    def forward(self, x, y):
        return x+y

class GAU_Block(nn.Module):
    def __init__(self, in_feats, r=16):
        super().__init__()
        # 参考PAN x 为浅层网络，y为深层网络
        self.x_conv = nn.Sequential(nn.Conv2d(in_feats, in_feats, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_feats))

        self.y_gap = nn.AdaptiveAvgPool2d(1)
        self.y_conv = nn.Sequential(nn.Conv2d(in_feats, in_feats, kernel_size=1, padding=0, bias=False),
                                    nn.BatchNorm2d(in_feats),
                                    nn.ReLU(inplace=True))

    def forward(self, y, x):
        x1 = self.x_conv(x)      # [B, c, h, w]

        y1 = self.y_gap(y)       # [B, c, 1, 1]
        y1 = self.y_conv(y1)     # [B, c, 1, 1]

        out = y1*x1 + y
        return out

class CBR(nn.Module):
    def __init__(self, in_feats, out_feats, k=2):
        super().__init__()
        self.mid_cbr = nn.Sequential(nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(in_feats), nn.ReLU(inplace=True)) if k != 1 else None
        self.out_cbr = nn.Sequential(nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_feats), nn.ReLU(inplace=True))
        
    def forward(self, x):
        if self.mid_cbr is not None:
            x = self.mid_cbr(x)
        return self.out_cbr(x)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, upsample=False):
        super(BasicBlock, self).__init__()
        self.upsample = upsample
        self.CBR1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(planes),
                                  nn.ReLU(inplace=True))

        if not upsample:
            self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(planes))
        else:
            self.conv2=nn.Sequential(nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=2, padding=0, bias=False),
                                     nn.BatchNorm2d(planes))
            self.up = nn.Sequential(nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=2, padding=0, bias=False),
                                    nn.BatchNorm2d(planes))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.upsample:
            identity = self.up(x)
        else:
            identity = x

        out = self.CBR1(x)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)

        return out

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

class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MRF_Concat_5_2, *shapes)