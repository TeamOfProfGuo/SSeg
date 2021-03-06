import sys
from addict import Dict
from copy import deepcopy

def get_config(exp_id):
    date, mode, idx = tuple(exp_id.split('_'))
    config = build_config(TEMPLATE, idx)
    if date in ('0821', '0822'):
        config.training.lr_setting = 'final_v%s' % mode[0]
        config.decoder_args.final_aux = (mode[1] == 't')
    elif date in ('0823', '0824', '0825', '0826', '0903', '0907'):
        dataset_dict = {'n': 'nyud', 's': 'sunrgbd'}
        config.training.dataset = dataset_dict[mode[0]]
        config.training.lr_setting = 'final_v2'
        config.decoder_args.final_aux = False
        if date == '0903':
            config.encoder_args.fuse_module = 'fuse'
            config.training.export = True
            config.training.class_weight = None
            config.decoder_args.aux = False
        if date == '0907':
            config.training.class_weight = None
            config.decoder_args.aux = False
        if len(mode) == 1:
            pass
        elif len(mode) == 2:
            config.training.epochs = int(mode[1]) * 100
        elif len(mode) > 2:
            encoder_dict = {'b': '2b', 'c': '2c'}
            epoch_dict = {
                't':  10, '0':  80, '1': 100, 'a': 150, '2': 200, 
                'b': 250, '3': 300, '4': 400, '5': 500, '6': 600, 
            }
            config.training.epochs = epoch_dict[mode[1]]
            config.training.lr = round(int(mode[2]) * 0.001, 3)
            config.training.aux_weight = round(int(mode[3]) * 0.1, 1)
            config.general.encoder = encoder_dict[mode[4]]
        else:
            raise ValueError('Invalid mode: %s.' % mode)
    elif date == 'test':
        dataset_dict = {'n': 'nyud', 's': 'sunrgbd'}
        config.training.dataset = dataset_dict[mode[0]]
        config.training.lr_setting = 'final_v2'
        config.training.epochs = 3
        config.decoder_args.final_aux = False
    elif date in ('nyud', 'sunrgbd'):
        encoder_dict = {'a': '2b', 'b': '2b', 'c': '2c'}
        config.training.dataset = date
        config.general.encoder = encoder_dict[mode]
        if mode == 'a':
            config.encoder_args.fuse_module = 'fuse'
            config.decoder_args.aux = False
    else:
        raise ValueError('Invalid Config ID: %s.' % exp_id)
    return config

def build_config(template_dict, idx):
    config = deepcopy(template_dict)
    config['decoder_args']['lf_args'].update(DECODER_ARGS[idx[0]])
    config['encoder_args']['fuse_args'] = FUSE_ARGS[idx[1]]
    config['decoder_args']['lf_args']['fuse_args'] = FUSE_ARGS[idx[2]]
    return Dict(config)

def test_config():
    config = get_config(sys.argv[1])
    for k, v in config.items():
        print('[%s]: %s' % (k, v))

TEMPLATE = {
    'training': {
        # Dataset
        'dataset': 'nyud',
        'workers': 4,
        'base_size': 520,
        'crop_size': 480,
        'train_split': 'train',
        # Util
        'early_fusion': False,
        'export': False,
        # Aux loss
        # 'aux': True,
        'aux_weight': 0.5,
        'class_weight': 1,
        # Training setting
        'epochs': 600,
        'start_epoch': 0,
        'batch_size': 8,
        'test_batch_size': 8,
        'lr': 0.003,
        'lr_setting': 'ori',
        'lr_scheduler': 'poly',
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'use_cuda': True,
        'seed': 1,
        # Checkpoint
        'resume': None,
        'checkname': 'default',
        'model_zoo': None,
        'ft': False
    },
    'general': {
        'encoder': '2b',
        'cp': 'none',
        'decoder': 'base',
        'feats': 'x'
    },
    'encoder_args': {
        'fuse_args': {},
        'pass_rff': (True, False),
        'fuse_module': 'psk'
    },
    'cp_args': {

    },
    'decoder_args': {
        'aux': True,
        'final_aux': False,
        'final_fuse': 'none',
        'lf_args': {
            'conv_flag': None,
            'lf_bb': None,
            'fuse_args': {},
            'fuse_module': 'fuse'
        },
        'final_args': {}
    }
}

DECODER_ARGS = {
    'a': {
        'conv_flag': (False, False),
        'lf_bb': 'none'
    },
    'b': {
        'conv_flag': (True, False),
        'lf_bb': 'rbb[2->2]'
    },
    'c': {
        'conv_flag': (True, False),
        'lf_bb': 'irb[2->2]'
    }
}

FUSE_ARGS = {
    # PSK (mmf only)
    'x': {
        'sp': 'x',
        'mid_feats': 16,
        'act_fn': 'sigmoid'
    },
    'y': {
        'sp': 'y',
        'mid_feats': 16,
        'act_fn': 'sigmoid'
    },
    'z': {
        'sp': 'u',
        'mid_feats': 16,
        'act_fn': 'sigmoid'
    },
    '1': {
        'sp': 'u',
        'mid_feats': 16,
        'act_fn': 'sigmoid',
        'pp_size': (1, )
    },
    '2': {
        'sp': 'u',
        'mid_feats': 16,
        'act_fn': 'sigmoid',
        'pp_size': (1, 2)
    },
    '3': {
        'sp': 'u',
        'mid_feats': 16,
        'act_fn': 'sigmoid',
        'pp_size': (1, 2, 4)
    },
    '4': {
        'sp': 'u',
        'mid_feats': 16,
        'act_fn': 'sigmoid',
        'pp_size': (1, 2, 4, 8)
    },
    '5': {
        'sp': 'u',
        'mid_feats': 16,
        'act_fn': 'sigmoid',
        'pp_size': (1, ),
        'descriptor': -1
    },
    '6': {
        'sp': 'u',
        'mid_feats': 16,
        'act_fn': 'sigmoid',
        'pp_size': (1, 2),
        'descriptor': -1
    },
    '7': {
        'sp': 'u',
        'mid_feats': 16,
        'act_fn': 'sigmoid',
        'pp_size': (1, 2, 4),
        'descriptor': -1
    },
    '8': {
        'sp': 'u',
        'mid_feats': 16,
        'act_fn': 'sigmoid',
        'pp_size': (1, 2, 4, 8),
        'descriptor': -1
    },
    # GF (4)
    'g': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gc1',
            'init': [False, True],
            'civ': 0.5
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    'h': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gc2',
            'init': [False, True],
            'civ': 0.5
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    'i': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'add',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    # simple add
    'a': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'add',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'idt',
        'att_setting': {}
    },
}

if __name__ == '__main__':
    test_config()