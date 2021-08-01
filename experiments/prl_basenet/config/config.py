import sys
from addict import Dict
from copy import deepcopy

def get_config(exp_id):
    date, idx = tuple(exp_id.split('_'))
    if date == '0801':
        if idx[0] == 'c':
            idx = 'a' + idx[1:]
        elif idx[0] == 'd':
            idx = 'b' + idx[1:]
        elif idx[0] == 'e':
            idx = 'c' + idx[1:]
        else:
            raise ValueError('Invalid Config ID: %s.' % exp_id)
        return get_gcgf(GCGF_TEMPLATE, GCGF_ARGS[idx], DECODER_ARGS[1])
    else:
        raise ValueError('Invalid Config ID: %s.' % exp_id)

def get_gcgf(template_dict, fuse_args, decoder_args):
    config = deepcopy(template_dict)
    config['fuse_args']['gcgf_args'] = fuse_args
    config['decoder_args'].update(decoder_args)
    config['decoder_args']['lf_args'] = fuse_args
    return Dict(config)

def test_config():
    config = get_config(sys.argv[1])
    for k, v in config.items():
        print('[%s]: %s' % (k, v))

GCGF_TEMPLATE = {
    'general': {
        'ef': False,
        'decoder': 'base',
        'n_features': None,
        'rgbd_fuse': 'gcgf',
        'sep_fuse': False,
        'cp': 'none'
    },
    'fuse_args': {
        'gcgf_args': {}
    },
    'cp_args': {},
    'decoder_feat': {
        'level': [256, 128, 64],
        'final': None
    },
    'decoder_args': {
        # 'conv_module': 'ctmd irb[4->4] irb[4->4] irb[4->2] luu(2)',
        'level_fuse': 'gcgf', 
        'feats': 'f', 
        # 'rf_conv': [True, False], 
        # 'lf_bb': 'rbb[2->2]', 
        'lf_args': {}
    }
}

DECODER_ARGS = {
    1: {
        'conv_module': 'ctmd irb[4->4] irb[4->4] irb[4->2] luu(2)',
        'rf_conv': [False, False],
        'lf_bb': 'none'
    },
    2: {
        'conv_module': 'ctmd irb[4->4] irb[4->4] irb[4->2] luu(2)',
        'rf_conv': [True, False],
        'lf_bb': 'rbb[2->2]'
    },
    3: {
        'conv_module': 'ctmd irb[4->4] irb[4->4] irb[4->2] luu(2)',
        'rf_conv': [True, False],
        'lf_bb': 'irb[2->2]'
    },
    4: {
        'conv_module': 'rbb7o',
        'rf_conv': [False, False],
        'lf_bb': 'none'
    },
    5: {
        'conv_module': 'rbb7o',
        'rf_conv': [True, False],
        'lf_bb': 'rbb[2->2]'
    },
    6: {
        'conv_module': 'rbb7o',
        'rf_conv': [True, False],
        'lf_bb': 'irb[2->2]'
    },
}

GCGF_ARGS = {
    'a1': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'add',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    'a2': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'cc3',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    'a3': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 1
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    'a4': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 0.5
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    'a5': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'add',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    'a6': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'cc3',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    'a7': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 1
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    'a8': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 0.5
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    'a9': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'add',
            'init': [True, False],
            'civ': None
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    'a10': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'cc3',
            'init': [True, False],
            'civ': None
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    'a11': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'gcgf',
            'init': [True, True],
            'civ': 1
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    'a12': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'gcgf',
            'init': [True, True],
            'civ': 0.5
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    'b1': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'add',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'se',
        'att_setting': {}
    },
    'b2': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'cc3',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'se',
        'att_setting': {}
    },
    'b3': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 1
        },
        'att_module': 'se',
        'att_setting': {}
    },
    'b4': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 0.5
        },
        'att_module': 'se',
        'att_setting': {}
    },
    'b5': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'add',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'se',
        'att_setting': {}
    },
    'b6': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'cc3',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'se',
        'att_setting': {}
    },
    'b7': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 1
        },
        'att_module': 'se',
        'att_setting': {}
    },
    'b8': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 0.5
        },
        'att_module': 'se',
        'att_setting': {}
    },
    'b9': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'add',
            'init': [True, False],
            'civ': None
        },
        'att_module': 'se',
        'att_setting': {}
    },
    'b10': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'cc3',
            'init': [True, False],
            'civ': None
        },
        'att_module': 'se',
        'att_setting': {}
    },
    'b11': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'gcgf',
            'init': [True, True],
            'civ': 1
        },
        'att_module': 'se',
        'att_setting': {}
    },
    'b12': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'gcgf',
            'init': [True, True],
            'civ': 0.5
        },
        'att_module': 'se',
        'att_setting': {}
    },
    'c1': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'add',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    'c2': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'cc3',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    'c3': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 1
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    'c4': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 0.5
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    'c5': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'add',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    'c6': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'cc3',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    'c7': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 1
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    'c8': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 0.5
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    'c9': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'add',
            'init': [True, False],
            'civ': None
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    'c10': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'cc3',
            'init': [True, False],
            'civ': None
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    'c11': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'gcgf',
            'init': [True, True],
            'civ': 1
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    'c12': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'gcgf',
            'init': [True, True],
            'civ': 0.5
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
}

if __name__ == '__main__':
    test_config()