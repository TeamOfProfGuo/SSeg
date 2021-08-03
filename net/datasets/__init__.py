from .nyud_v2 import NYUD
from .sun_rgbd import SUNRGBD
from .nyud_v2_tmp import NYUD_tmp

datasets = {
    'nyud': NYUD,
    'sunrgbd': SUNRGBD,
    'nyud_tmp': NYUD_tmp
}

def get_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)