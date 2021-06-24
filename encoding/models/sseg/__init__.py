from .base import *
from .fcn import *
from .psp import *
from .fcfpn import *
from .atten import *
from .encnet import *
from .deeplab import *
from .upernet import *
from .fusenet import *
from .danet import *
from .danet_dep import *
from .danet_d import *
from .ppanet import *
from .lanet import *
from .danet_psp import *     # process depth using PSP module, then concat it with RGB as main input
from .unet import *
from .linknet import *
from .munet import *
from .lmfnet import *
from .refinenet import *

def get_segmentation_model(name, **kwargs):
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'fcfpn': get_fcfpn,
        'atten': get_atten,
        'encnet': get_encnet,
        'upernet': get_upernet,
        'deeplab': get_deeplab,
        'pspd': get_pspd,
        'fuse': get_fuse,
        'danet': get_danet,
        'danet_dep': get_danet_dep,
        'danet_d': get_danet_d,
        'ppa': get_ppanet,
        'lanet': get_lanet,
        'danet_psp': get_danet_psp,
        'unet': get_Unet,
        'att_unet': get_AttUnet,
        'mfnet': get_mfnet,
        'linknet': get_linknet,
        'munet': get_munet,
        'attlinknet': get_attlinknet,
        'link_mfnet': get_LinkMFNet,
        'refinenet': get_refinenet
    }
    return models[name.lower()](**kwargs)
