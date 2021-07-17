
from torch.nn import functional as F

__all__ = ['interpolate']

def interpolate(x, size, mode = 'bilinear'):
    if mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
        return F.interpolate(x, size=size, mode=mode, align_corners=True)
    else:
        return F.interpolate(x, size=size, mode=mode)