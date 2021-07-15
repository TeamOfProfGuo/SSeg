from .basenet import *
from .refinenet import *

def get_segmentation_model(name, **kwargs):
    models = {
        'refinenet': get_refinenet,
        'basenet': get_basenet
    }
    return models[name.lower()](**kwargs)