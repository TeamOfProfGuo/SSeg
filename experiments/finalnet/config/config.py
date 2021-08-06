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
        return get_gcgf(GCGF_TEMPLATE, GCGF_ARGS_0801[idx], DECODER_ARGS[1])
    elif date in ('0802', '0804', '0805'):
        decoder_dict = {'a': 1, 'b': 2, 'c': 3}
        decoder_idx = decoder_dict[idx[0]]
        fuse1_idx = int(idx[1])
        fuse2_idx = int(idx[2])
        return get_gcgf_diff(
            template_dict=GCGF_TEMPLATE,
            fuse_args1=GCGF_ARGS[fuse1_idx],
            fuse_args2=GCGF_ARGS[fuse2_idx],
            decoder_args=DECODER_ARGS[decoder_idx]
        )
    elif date in ('0806', 'TBA'):
        decoder_dict = {'a': 7, 'b': 8, 'c': 9}
        decoder_idx = decoder_dict[idx[0]]
        fuse1_idx = int(idx[1])
        fuse2_idx = int(idx[2])
        use_aux = (idx[3] == 't')
        return get_gcgf_diff(
            template_dict=GCGF_TEMPLATE,
            fuse_args1=GCGF_ARGS[fuse1_idx],
            fuse_args2=GCGF_ARGS[fuse2_idx],
            decoder_args=DECODER_ARGS[decoder_idx],
            aux = use_aux
        )
    else:
        raise ValueError('Invalid Config ID: %s.' % exp_id)

def get_gcgf(template_dict, fuse_args, decoder_args):
    config = deepcopy(template_dict)
    config['fuse_args']['gcgf_args'] = fuse_args
    config['decoder_args'].update(decoder_args)
    config['decoder_args']['lf_args'] = fuse_args
    return Dict(config)

def get_gcgf_diff(template_dict, fuse_args1, fuse_args2, decoder_args, aux=False):
    config = deepcopy(template_dict)
    config['fuse_args']['gcgf_args'] = fuse_args1
    config['decoder_args']['aux'] = aux
    config['decoder_args'].update(decoder_args)
    config['decoder_args']['lf_args'] = fuse_args2
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
        'aux': False,
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
    7: {
        'conv_module': 'irb',
        'rf_conv': [False, False],
        'lf_bb': 'none'
    },
    8: {
        'conv_module': 'irb',
        'rf_conv': [True, False],
        'lf_bb': 'rbb[2->2]'
    },
    9: {
        'conv_module': 'irb',
        'rf_conv': [True, False],
        'lf_bb': 'irb[2->2]'
    },
}

GCGF_ARGS = {
    1: {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'add',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    2: {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'la',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    3: {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 1
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    4: {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 0.5
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    5: {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'add',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    6: {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'add',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    7: {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 1
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    8: {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 0.5
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    0: {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': -1
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    9: {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': -1
        },
        'att_module': 'idt',
        'att_setting': {}
    }
}

GCGF_ARGS_0801 = {
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