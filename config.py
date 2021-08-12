import sys
from addict import Dict
from copy import deepcopy

def get_config(exp_id):
    date, idx = tuple(exp_id.split('_'))
    if date in ('0809', 'test'):
        encoder_dict = {'2': '2b', '3': '3b'}
        encoder_branches = encoder_dict[idx[0]]
        decoder_feats = idx[1]
        decoder_idx = idx[2]
        fuse1_idx = idx[3]
        fuse2_idx = idx[4]
        use_aux = (idx[5] == 't')
        return build_config(
            template_dict=TEMPLATE,
            encoder=encoder_branches,
            fuse_args1=FUSE_ARGS[fuse1_idx],
            fuse_args2=FUSE_ARGS[fuse2_idx],
            decoder_feats=decoder_feats,
            lf_args=DECODER_ARGS[decoder_idx],
            aux = use_aux
        )
    elif date.find('0810') != -1:
        if 'a' in date:
            config = get_2bxt_config(idx)
            config.training.lr_setting = 'ori'
        elif 'b' in date:
            config = get_2bxt_config(idx)
            config.training.lr_setting = 'same-2b'
        elif 'c' in date:
            config = get_2bxt_config(idx)
            config.general.cp = 'psp'
            if date[-1] == '1':
                config.cp_args = {'size': (1, 2, 3, 6)}
            elif date[-1] == '2':
                config.cp_args = {'size': (1, 2, 4, 8)}
            elif date[-1] == '3':
                config.cp_args = {'size': (1, 3, 5, 7)}
            elif date[-1] == '4':
                config.cp_args = {'size': (1, 3, 5, 8)}
            else:
                raise ValueError('Invalid cp_args: %s.' % date)
        elif 'd' in date:
            config = build_config(
                template_dict=TEMPLATE,
                encoder='3b',
                fuse_args1=FUSE_ARGS[idx[1]],
                fuse_args2=FUSE_ARGS[idx[2]],
                decoder_feats='z',
                lf_args=DECODER_ARGS[idx[0]],
                aux = (idx[3] == 't')
            )
            config.training.lr_setting = 'same-3b'
            config.encoder_args.pass_rff = (False, False, True)
        return config
    elif date.find('0811') != -1:
        if 'a' in date:
            config = get_2bxt_config(idx)
            aux_weight = {'1': 0.2, '2': 0.4, '3': 0.6, '4': 0.8}
            config.training.aux_weight = aux_weight[idx[-1]]
        elif 'b' in date:
            config = get_2bxt_config(idx)
            config.decoder_args.lf_args.fuse_module = 'ca6'
        return config
    elif date.find('0812') != -1:
        if 'a' in date:
            config = get_2bxt_config(idx)
            config.training.aux_weight = 0.5
        elif 'b' in date:
            config = get_2bxt_config(idx)
            aux_weight = {'1': 0.3, '2': 0.5, '3': 0.7}
            config.training.aux_weight = aux_weight[idx[-1]]
            config.decoder_args.lf_args.fuse_module = 'ca6'
        elif 'c' in date:
            config = get_2bxt_config(idx)
            config.training.aux_weight = 0.5
            if idx[-1] == 'o':
                config.decoder_args.final_fuse = 'apnb'
                config.decoder_args.final_args = {
                    'in_channels': 64,
                    'out_channels': 64,
                    'key_channels': 64,
                    'value_channels': 64
                }
        elif 'd' in date:
            config = get_2bxt_config(idx)
            config.training.aux_weight = 0.5
            config.decoder_args.lf_args.fuse_module = 'pa0'
        elif 'e' in date:
            config = get_2bxt_config(idx)
            config.training.aux_weight = 0.5
            config.encoder_args.fuse_module = 'ca2b'
        return config
    else:
        raise ValueError('Invalid Config ID: %s.' % exp_id)

def build_config(template_dict, encoder, fuse_args1, fuse_args2, 
                 decoder_feats, lf_args, aux=False, final_fuse='none'):
    config = deepcopy(template_dict)
    config['general']['encoder'] = encoder
    config['general']['feats'] = decoder_feats
    config['encoder_args']['fuse_args'] = fuse_args1
    config['decoder_args']['aux'] = aux
    config['decoder_args']['final_fuse'] = final_fuse
    config['decoder_args']['lf_args'].update(lf_args)
    config['decoder_args']['lf_args']['fuse_args'] = fuse_args2
    return Dict(config)

def get_2bxt_config(idx):
    return build_config(
        template_dict=TEMPLATE,
        encoder='2b',
        fuse_args1=FUSE_ARGS[idx[1]],
        fuse_args2=FUSE_ARGS[idx[2]],
        decoder_feats='x',
        lf_args=DECODER_ARGS[idx[0]],
        aux = (idx[3] == 't')
    )

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
        'aux': None,
        'aux_weight': 0.2,
        'se_loss': False,
        'se_weight': 0.2,
        # Training setting
        'epochs': 600,
        'start_epoch': 0,
        'batch_size': 8,
        'test_batch_size': 8,
        'lr': 0.001,
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
        'fuse_module': 'fuse'
    },
    'cp_args': {

    },
    'decoder_args': {
        'aux': False,
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
    # General Fusion Module
    '1': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'add',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    '2': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'la',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    '3': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 1
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    '4': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 0.5
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    '5': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'add',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    '6': {
        'fuse_setting': {
            'pre_bn': True,
            'merge': 'add',
            'init': [False, False],
            'civ': None
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    '7': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 1
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    '8': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': 0.5
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    '0': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': -1
        },
        'att_module': 'pdl',
        'att_setting': {}
    },
    '9': {
        'fuse_setting': {
            'pre_bn': False,
            'merge': 'gcgf',
            'init': [False, True],
            'civ': -1
        },
        'att_module': 'idt',
        'att_setting': {}
    },
    # CA6 Module (mrf only)
    'i': {
        'act_fn': 'idt',
        'pass_rff': False
    },
    't': {
        'act_fn': 'tanh',
        'pass_rff': False
    },
    's': {
        'act_fn': 'sigmoid',
        'pass_rff': False
    },
    # PA0 Module (mrf only)
    'p': {
        'act_fn': 'sigmoid'
    },
    'q': {
        'act_fn': 'tanh'
    },
    # CA2b  (mmf only)
    'x': {
        'act_fn': 'sigmoid'
    },
    'y': {
        'act_fn': 'softmax'
    },
}

if __name__ == '__main__':
    test_config()