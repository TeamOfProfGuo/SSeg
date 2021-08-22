
import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import init_conv

class Fuse_Module(nn.Module):
    def __init__(self, in_feats, fuse_setting={}, att_module='idt', att_setting={}):
        super().__init__()
        module_dict = {
            'idt': IDT_Block,
            'se': SE_Block,
            'pdl': PDL_Block
        }
        self.att_module = att_module
        self.pre1 = module_dict[att_module](in_feats, **att_setting)
        self.pre2 = module_dict[att_module](in_feats, **att_setting)
        self.gcgf = General_Fuse_Block(in_feats, **fuse_setting)
    
    def forward(self, x, y):
        if self.att_module != 'idt':
            x = self.pre1(x)
            y = self.pre2(y)
        return self.gcgf(x, y), x, y

class General_Fuse_Block(nn.Module):
    def __init__(self, in_feats, pre_bn=False, merge='gcgf', init=(False, True), civ=1):
        super().__init__()
        merge_dict = {
            'gc1': nn.Conv2d(2*in_feats, in_feats, kernel_size=1, padding=0, groups=in_feats, bias=True),
            'gc2': nn.Conv2d(2*in_feats, in_feats, kernel_size=1, padding=0, groups=in_feats, bias=False),
            'add': Add_Merge(in_feats),
            'cc3': CC3_Merge(in_feats),
            'la': LA_Merge(in_feats)
        }
        if pre_bn:
            self.pre_bn1 = nn.BatchNorm2d(in_feats)
            self.pre_bn2 = nn.BatchNorm2d(in_feats)
        self.pre_bn = pre_bn
        self.feats = in_feats
        self.merge_mode = merge
        self.merge = merge_dict[merge]
        self._init_weights(init, civ)
        
    def forward(self, x, y):
        b, c, h, w = x.size()
        if self.pre_bn:
            x = self.pre_bn1(x)
            y = self.pre_bn2(y)
        if 'gc' not in self.merge_mode:
            return self.merge(x, y)
        feats = torch.cat((x, y), dim=-2).reshape(b, 2*c, h, w)   # [b, c, 2h, w] => [b, 2c, h, w]
        return self.merge(feats)
            
    def _init_weights(self, init, civ):
        if init[0] and self.pre_bn:
            self.pre_bn1.weight.data.fill_(1)
            self.pre_bn2.weight.data.fill_(1)
            self.pre_bn1.bias.data.zero_()
            self.pre_bn2.bias.data.zero_()
        if init[1] and isinstance(self.merge, nn.Conv2d):
            if civ == -1:
                self.merge.weight.data = init_conv(self.feats, 2, 1, 'b')
            else:
                self.merge.weight.data.fill_(civ)

class CA6_Module(nn.Module):
    def __init__(self, in_feats, act_fn='idt', pass_rff=False):
        super().__init__()
        # 参考PAN x 为浅层网络，y为深层网络
        act_dict = {
            'idt': nn.Identity,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }
        self.pass_rff = pass_rff
        self.x_conv = nn.Sequential(
            nn.Conv2d(in_feats, in_feats, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_feats)
        )
        self.y_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_feats, in_feats, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_feats),
            nn.ReLU(inplace=True),
            act_dict[act_fn]()
        )

    def forward(self, y, x):
        z = self.x_conv(x)      # [B, c, h, w]
        w = self.y_conv(y)      # [B, c, 1, 1]
        return (w * z + y), None, (z if self.pass_rff else x)

class PA0_Module(nn.Module):
    def __init__(self, ch, r=4, act_fn='sigmoid'):
        super().__init__()

        int_ch = max(ch//r, 32)
        self.act_fn = act_fn
        if act_fn == 'sigmoid':
            self.conv, self.fuse = 'conv', 'cat'
        elif act_fn == 'tanh':
            self.conv, self.fuse = 'conv', 'add'

        self.W_x = nn.Sequential(
            nn.Conv2d(ch, int_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(int_ch)
        )
        self.W_y = nn.Sequential(
            nn.Conv2d(ch, int_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(int_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(int_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1)
        )
        self.relu = nn.ReLU(inplace=True)

        if self.conv == 'conv':
            self.x_conv = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch)
            )
        if self.fuse == 'cat':
            self.out_conv = nn.Conv2d(ch * 2, ch, kernel_size=1, stride=1)

    def forward(self, y, x):   # 对x(第二个param)进行attention处理 y深层网络 x浅层网络
        x1 = self.W_x(x)           # [B, int_c, h, w]
        y1 = self.W_y(y)           # [B, int_c, h, w]
        psi = self.relu(x1 + y1)   # no bias
        psi = self.psi(psi)        # [B, 1, h, w]

        if self.act_fn == 'sigmoid':
            psi = F.sigmoid(psi)
        elif self.act_fn == 'rsigmoid':
            psi = F.sigmoid(F.relu(psi, inplace=True))
        elif self.act_fn == 'tanh':
            psi = F.tanh(F.relu(psi, inplace=True))

        if self.conv:
            x = self.x_conv(x)
        weighted_x = x * psi

        if self.fuse == 'add':
            return (weighted_x + y), None, weighted_x
        elif self.fuse == 'cat':
            return self.out_conv(torch.cat((weighted_x, y), dim=1)), None, weighted_x

class CA2b_Module(nn.Module):
    def __init__(self, in_ch, r=16, act_fn=None):
        """ Attention as in SKNet (selective kernel) """
        super().__init__()
        self.act_fn = act_fn
        self.pp_size = (1, 3)
        d = max(int(in_ch/r), 32)
        #d = 32 if shape[0] >= 30 else 16
        pp_d = sum(e**2 for e in self.pp_size)
        print('pp_size: {} dimension d {}'.format(self.pp_size, d))

        # to calculate Z
        self.fc = nn.Sequential(nn.Linear(in_ch*pp_d, d, bias=False),
                                nn.BatchNorm1d(d),
                                nn.ReLU(inplace=True))
        # 各个分支
        self.fc_x = nn.Linear(d, in_ch)
        self.fc_y = nn.Linear(d, in_ch)
        if act_fn == 'sigmoid':
            self.act_x = nn.Sigmoid()
            self.act_y = nn.Sigmoid()
        elif act_fn == 'tanh':
            self.act_x = nn.Sequential(nn.ReLU(inplace=True), nn.Tanh())
            self.act_y = nn.Sequential(nn.ReLU(inplace=True), nn.Tanh())
        elif act_fn == 'rsigmoid':
            self.act_x = nn.Sequential(nn.ReLU(inplace=True), nn.Sigmoid())
            self.act_y = nn.Sequential(nn.ReLU(inplace=True), nn.Sigmoid())
        elif act_fn == 'softmax':
            self.act = nn.Softmax(dim=1)

    def forward(self, x, y):
        d = y.clone()
        U = x+y
        batch_size, ch, _, _ = U.size()

        ppool = []
        for s in self.pp_size:
            ppool.append(F.adaptive_avg_pool2d(U, s).view(batch_size, ch, -1))  # [B, c, s*s]
        z = torch.cat(tuple(ppool), dim=-1)            # [B, c, 1+9+25]
        z = z.view(batch_size, -1).contiguous()        # [B, c*35]
        z = self.fc(z)                                 # [B, d]

        z_x = self.fc_x(z)  # [B, c]
        z_y = self.fc_y(z)  # [B, c]
        if self.act_fn in ['sigmoid', 'tanh', 'rsigmoid']:
            w_x = self.act_x(z_x)    # [B, c]
            w_y = self.act_y(z_y)    # [B, c]
        elif self.act_fn == 'softmax':
            w_xy = torch.cat((z_x, z_y), dim=1)    # [B, 2c]
            w_xy = w_xy.view(batch_size, 2, ch)    # [B, 2, c]
            w_xy = self.act(w_xy)                  # [B, 2, c]
            w_x, w_y = w_xy[:, 0].contiguous(), w_xy[:, 1].contiguous()      # [B, c]
        out = x * w_x.view(batch_size, ch, 1, 1) + y * w_y.view(batch_size, ch, 1, 1)
        return out, None, d

class PSK_Module(nn.Module):
    def __init__(self, in_feats, pp_size=(1, 2, 4, 8), descriptor=8, mid_feats=32, act_fn=None, sp='x'):
        super().__init__()

        print('sp = %s, pp = %s, dd = %d, m = %d, act = %s.' % (sp, pp_size, descriptor, mid_feats, act_fn))
        self.sp = sp
        self.pp_size = pp_size
        self.feats_size = sum([(s ** 2) for s in self.pp_size])
        self.descriptor = descriptor
        self.act_fn = act_fn

        self.des = nn.Conv2d(self.feats_size, self.descriptor, kernel_size=1)
        self.fc = nn.Sequential(nn.Linear(in_feats * descriptor, mid_feats, bias=False),
                                nn.BatchNorm1d(mid_feats),
                                nn.ReLU(inplace=True))

        self.fc_x = nn.Linear(mid_feats, in_feats)
        self.fc_y = nn.Linear(mid_feats, in_feats)
        if act_fn in ('sigmoid', 'sig'):
            self.act_x = nn.Sigmoid()
            self.act_y = nn.Sigmoid()
        elif act_fn in ('rsigmoid', 'rsig'):
            self.act_x = nn.Sequential(nn.ReLU(inplace=True), nn.Sigmoid())
            self.act_y = nn.Sequential(nn.ReLU(inplace=True), nn.Sigmoid())
        elif act_fn in ('softmax', 'soft'):
            self.act = nn.Softmax(dim=1)

    def forward(self, x, y):
        U = x + y
        batch_size, ch, _, _ = x.size()
        sp_dict = {'x': x, 'y': y, 'u': U}

        pooling_pyramid = []
        for s in self.pp_size:
            pooling_pyramid.append(F.adaptive_avg_pool2d(sp_dict[self.sp], s).view(batch_size, ch, 1, -1))  # [b, c, 1, s^2]
        z = torch.cat(tuple(pooling_pyramid), dim=-1)           # [b, c, 1, f]
        z = z.reshape(batch_size * ch, -1, 1, 1)                # [bc, f, 1, 1]
        z = self.des(z).view(batch_size, ch * self.descriptor)  # [bc, d, 1, 1] => [b, cd]
        z = self.fc(z)      # [b, m]

        z_x = self.fc_x(z)  # [b, c]
        z_y = self.fc_y(z)  # [b, c]
        if self.act_fn in ['sigmoid','rsigmoid', 'sig', 'rsig']:
            w_x = self.act_x(z_x)  # [b, c]
            w_y = self.act_y(z_y)  # [b, c]
        elif self.act_fn in ['softmax', 'soft']:
            w_xy = torch.cat((z_x, z_y), dim=1)  # [b, 2c]
            w_xy = w_xy.view(batch_size, 2, ch)  # [b, 2, c]
            w_xy = self.act(w_xy)                # [b, 2, c]
            w_x, w_y = w_xy[:, 0].contiguous(), w_xy[:, 1].contiguous()  # [b, c]
        rf_x = x * w_x.view(batch_size, ch, 1, 1)
        rf_y = y * w_y.view(batch_size, ch, 1, 1)
        out = rf_x + rf_y
        return out, rf_x, rf_y 

class Merge_Module(nn.Module):
    def __init__(self, in_feats, fuse_setting={}, att_module='idt', att_setting={}):
        super().__init__()
        module_dict = {
            'idt': IDT_Block,
            'se': SE_Block,
            'pdl': PDL_Block
        }
        self.att_module = att_module
        self.pre1 = module_dict[att_module](in_feats, **att_setting)
        self.pre2 = module_dict[att_module](in_feats, **att_setting)
        self.gcgf = General_Merge_Block(in_feats, **fuse_setting)
    
    def forward(self, m, x, y):
        if self.att_module != 'idt':
            x = self.pre1(x)
            y = self.pre2(y)
        return self.gcgf(m, x, y), x, y

class General_Merge_Block(nn.Module):
    def __init__(self, in_feats, pre_bn=False, merge='gcgf', init=(False, True), civ=-1):
        super().__init__()
        if pre_bn:
            self.pre_bn1 = nn.BatchNorm2d(in_feats)
            self.pre_bn2 = nn.BatchNorm2d(in_feats)
        self.pre_bn = pre_bn
        self.feats = in_feats
        self.merge = nn.Conv2d(3*in_feats, in_feats, kernel_size=1, padding=0, groups=in_feats, bias=True)
        self._init_weights(init, civ)
        
    def forward(self, m, x, y):
        b, c, h, w = x.size()
        if self.pre_bn:
            x = self.pre_bn1(x)
            y = self.pre_bn2(y)
        feats = torch.cat((m, x, y), dim=-2).reshape(b, 3*c, h, w)
        return self.merge(feats)
            
    def _init_weights(self, init, civ):
        if init[0] and self.pre_bn:
            self.pre_bn1.weight.data.fill_(1)
            self.pre_bn2.weight.data.fill_(1)
            self.pre_bn1.bias.data.zero_()
            self.pre_bn2.bias.data.zero_()
        if init[1] and isinstance(self.merge, nn.Conv2d):
            if civ == -1:
                self.merge.weight.data = init_conv(self.feats, 3, 1, 'b')
            else:
                self.merge.weight.data.fill_(civ)

# ARM Blocks

class SE_Block(nn.Module):
    def __init__(self, in_feats, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_feats, in_feats // r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_feats // r, in_feats, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(F.adaptive_avg_pool2d(x, 1))
        return w * x
    
class PDL_Block(nn.Module):
    def __init__(self, in_feats, pp_layer=4, descriptor=8, mid_feats=16):
        super().__init__()
        self.layer_size = pp_layer                  # l: pyramid layer num
        self.feats_size = (4 ** pp_layer - 1) // 3  # f: feats for descritor
        self.descriptor = descriptor                # d: descriptor num (for one channel)

        self.des = nn.Conv2d(self.feats_size, descriptor, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(descriptor * in_feats, mid_feats, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_feats, in_feats),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        l, f, d = self.layer_size, self.feats_size, self.descriptor
        pooling_pyramid = []
        for i in range(l):
            pooling_pyramid.append(F.adaptive_avg_pool2d(x, 2 ** i).view(b, c, 1, -1))
        y = torch.cat(tuple(pooling_pyramid), dim=-1)   # [b,  c, 1, f]
        y = y.reshape(b*c, f, 1, 1)                     # [bc, f, 1, 1]
        y = self.des(y).view(b, c*d)                    # [bc, d, 1, 1] => [b, cd, 1, 1]
        w = self.mlp(y).view(b, c, 1, 1)                # [b,  c, 1, 1] => [b, c, 1, 1]
        return w * x

class IDT_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

# Merge Blocks

class Add_Merge(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, x, y):
        return x+y

class LA_Merge(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lamb = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, y):
        return x + self.lamb * y

class CC3_Merge(nn.Module):
    def __init__(self, in_feats, *args, **kwargs):
        super().__init__()
        self.cc_block = nn.Sequential(
            nn.Conv2d(2*in_feats, in_feats, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, y):
        return self.cc_block(torch.cat((x, y), dim=1))

# Constant that stores available fusion modules
FUSE_MODULE_DICT = {
    'merge': Merge_Module,
    'fuse': Fuse_Module,
    'ca2b': CA2b_Module,
    'ca6': CA6_Module,
    'pa0': PA0_Module,
    'psk': PSK_Module
}