from .basenet import *
from .refinenet import *
from .prl_basenet import *

def get_segmentation_model(name, **kwargs):
    models = {
        # 'gfnet': get_gfnet,
        'basenet': get_basenet,
        'refinenet': get_refinenet,
        'prl_basenet': get_prl_basenet
    }
    return models[name.lower()](**kwargs)