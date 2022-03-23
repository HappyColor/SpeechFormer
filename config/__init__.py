
from .config import _C as Cfg
import os
import torch
import json

def create_workshop(cfg, local_rank):
    modeltype = cfg.model.type
    database = cfg.dataset.database
    batch = cfg.train.batch_size
    feature = cfg.dataset.feature
    lr = cfg.train.lr
    epoch = cfg.train.EPOCH
    
    world_size = torch.cuda.device_count()
    batch = batch * world_size

    if cfg.train.find_init_lr:
        if cfg.mark is None:
            cfg.mark = 'find_init_lr'
        else:
            cfg.mark = 'find_init_lr_' + cfg.mark
        config_name = f'./experiments/{modeltype}/{database}_b{batch}_{feature}'
    else:
        config_name = f'./experiments/{modeltype}/{database}_e{epoch}_b{batch}_lr{lr}_{feature}'

    if cfg.mark is not None:
        config_name = config_name + '_{}'.format(cfg.mark)

    cfg.workshop = os.path.join(config_name, f'fold_{cfg.train.current_fold}')
    cfg.ckpt_save_path = os.path.join(cfg.workshop, 'checkpoint')
    
    if local_rank == 0:
        if os.path.exists(cfg.workshop):
            raise ValueError(f'workshop {cfg.workshop} already existed.')
        else:
            os.makedirs(cfg.workshop)
            os.makedirs(cfg.ckpt_save_path)

def modify_config(cfg, args):
    # merge from Namespace
    args = vars(args)  # Namespace -> Dict
    args = dict_2_list(args)
    print('Use arguments from command line:', args)
    cfg.merge_from_list(args)

    modeltype = cfg.model.type
    database = cfg.dataset.database
    feature = cfg.dataset.feature
    
    train_config = json.load(open(f'./config/train_{modeltype}.json', 'r'))[database][feature]
    model_config = json.load(open('./config/model_config.json', 'r'))[modeltype]

    # merge from train_config
    train_config = dict_2_list(train_config)
    print('Use arguments from train json file:', train_config)
    cfg.merge_from_list(train_config)
    
    # modify cfg.mark
    if modeltype == 'Transformer':
        num_layers = model_config['num_layers']
        _mark = f'L{num_layers}'
    elif modeltype == 'SpeechFormer': 
        num_layers = model_config['num_layers']
        expand = model_config['expand']
        _mark = f'L{num_layers[0]}{num_layers[1]}{num_layers[2]}{num_layers[3]}_expa{expand[0]}{expand[1]}{expand[2]}'
    else:
        raise ValueError(f'Unknown model type: {modeltype}')

    cfg.mark = _mark + '_' + cfg.mark if cfg.mark is not None else _mark
    print('Modified mark:', cfg.mark)
    
    # add key: evaluate
    cfg.train.evaluate = json.load(open(f'./config/{database}_feature_config.json', 'r'))['evaluate']

def dict_2_list(dict):
    lst = []
    for key, value in dict.items():
        if value is not None:
            lst.extend([key, value])
    return lst
