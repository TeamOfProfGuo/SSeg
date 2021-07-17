# from .gfnet import *
from .basenet import *
from .refinenet import *

def get_segmentation_model(name, **kwargs):
    models = {
        # 'gfnet': get_gfnet,
        'basenet': get_basenet,
        'refinenet': get_refinenet
    }
    return models[name.lower()](**kwargs)