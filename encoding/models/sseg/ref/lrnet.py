
import torch


class GeometryPrior(torch.nn.Module):
    def __init__(self, k, channels, multiplier=0.5):
        super(GeometryPrior, self).__init__()
        self.channels = channels
        self.k = k
        self.position = 2 * torch.rand(1, 2, k, k, requires_grad=True) - 1      # [1, 2, k, k]
        self.l1 = torch.nn.Conv2d(2, int(multiplier * channels), 1)             # [1, inter-channel, k, k]
        self.l2 = torch.nn.Conv2d(int(multiplier * channels), channels, 1)      # [1, channel, k, k]

    def forward(self, x):
        x = self.l2(torch.nn.functional.relu(self.l1(self.position)))
        return x.view(1, self.channels, 1, self.k ** 2)                         # [1, channel, 1, kk]


class GeometryPrior(torch.nn.Module):
    def __init__(self, k, channels, multiplier=0.5):
        super(GeometryPrior, self).__init__()
        self.channels = channels
        self.k = k
        self.l1 = torch.nn.Conv2d(2, int(multiplier * channels), 1)
        self.l2 = torch.nn.Conv2d(int(multiplier * channels), channels, 1)

    def forward(self):
        # as the paper does not infer how to construct a [2,k,k] position matrix
        # we assume that it's a kxk matrix for delta-x,and a kxk matrix for delta-y.
        # that is, [[[-1,0,1],[-1,0,1],[-1,0,1]],[[1,1,1],[0,0,0],[-1,-1,-1]]] for kernel = 3
        a_range = torch.arange(-1 * (self.k // 2), (self.k // 2) + 1).view(1, -1)      # [1, k]
        x_position = a_range.expand(self.k, a_range.shape[1])                          # [k, k]
        b_range = torch.arange((self.k // 2), -1 * (self.k // 2) - 1, -1).view(-1, 1)  # [k, 1]
        y_position = b_range.expand(b_range.shape[0], self.k)                          # [k, k]
        position = torch.cat((x_position.unsqueeze(0), y_position.unsqueeze(0)), 0).unsqueeze(0).float()  # [1, 2, 7, 7]
        if torch.cuda.is_available():
            position = position.cuda()
        out = self.l2(torch.nn.functional.relu(self.l1(position)))     # [1, c, 7, 7]
        return out.view(1, self.channels, 1, self.k**2)



class KeyQueryMap(torch.nn.Module):
    def __init__(self, channels, m):
        super(KeyQueryMap, self).__init__()
        self.l = torch.nn.Conv2d(channels, channels // m, 1)

    def forward(self, x):
        return self.l(x)


class AppearanceComposability(torch.nn.Module):
    def __init__(self, k, padding, stride):
        super(AppearanceComposability, self).__init__()
        self.k = k
        self.unfold = torch.nn.Unfold(k, dilation=1, padding=padding, stride=stride)
        # Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)

    def forward(self, x):
        key_map, query_map = x                     # [B, c, h, w]
        k = self.k
        key_map_unfold = self.unfold(key_map)      # [B, kkc, L]
        query_map_unfold = self.unfold(query_map)  # [B, kkc, L]
        key_map_unfold = key_map_unfold.view(
            key_map.shape[0], key_map.shape[1], -1,
            key_map_unfold.shape[-2] // key_map.shape[1])      # [B, c, L, kk]
        query_map_unfold = query_map_unfold.view(
            query_map.shape[0], query_map.shape[1], -1,
            query_map_unfold.shape[-2] // query_map.shape[1])  # [B, c, L, kk]

        return key_map_unfold * query_map_unfold[:, :, :, k ** 2 // 2:k ** 2 // 2 + 1]  # [B, c, L, kk]


def combine_prior(appearance_kernel, geometry_kernel):
    return torch.nn.functional.softmax(appearance_kernel + geometry_kernel, dim=-1)


class LocalRelationalLayer(torch.nn.Module):
    def __init__(self, channels, k, stride=1, m=None, padding=0):
        super(LocalRelationalLayer, self).__init__()
        self.channels = channels
        self.k = k
        self.stride = stride
        self.m = m or 8
        self.padding = padding
        self.kmap = KeyQueryMap(channels, k)
        self.qmap = KeyQueryMap(channels, k)
        self.ac = AppearanceComposability(k, padding, stride)
        self.gp = GeometryPrior(k, channels // m)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)
        self.final1x1 = torch.nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # geometric kernel
        gpk = self.gp(0)         # geometric prior   #[1, c, 7, 7]  -> [1, c, 1, kk]
        # appearance kernel
        km = self.kmap(x)
        qm = self.qmap(x)
        ak = self.ac((km, qm))       # [B, c, L, kk]   # [B, c, hw, kk]

        ck = self.softmax(ak + gpk)  # [B, c, L, kk]: by channel attention

        x_unfold = self.unfold(x)    # [B, kkC, L]: value. x:[B, C, h, w]
        x_unfold = x_unfold.view(x.shape[0], self.m, x.shape[1] // self.m,       # [B, m, C/m, L, kk]
                                 -1, x_unfold.shape[-2] // x.shape[1])

        pre_output = ck * x_unfold   # [B, m, c, L, kk]
        pre_output = pre_output.view(x.shape[0], x.shape[1], -1, x_unfold.shape[-2] // x.shape[1])  # [B, C, L, kk]

        h_out = (x.shape[2] + 2 * self.padding - 1 * (self.k - 1) - 1) // self.stride + 1
        w_out = (x.shape[3] + 2 * self.padding - 1 * (self.k - 1) - 1) // self.stride + 1

        pre_output = torch.sum(pre_output, axis=-1).view(x.shape[0], x.shape[1], h_out, w_out)    # [B, C, h-out, w-out]
        return self.final1x1(pre_output)
