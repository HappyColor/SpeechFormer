
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

def modify_mark(cfg, filepath='./config/model_config.json'):
    modeltype = cfg.model.type
    
    if modeltype == 'Transformer':
        with open(filepath, 'r') as f:
            model_json = json.load(f)[modeltype]
            num_layers = model_json['num_layers']
        _mark = f'L{num_layers}'
    elif modeltype == 'SpeechFormer':
        with open(filepath, 'r') as f:
            model_json = json.load(f)[modeltype]
            num_layers = model_json['num_layers']
            expand = model_json['expand']
        _mark = f'L{num_layers[0]}{num_layers[1]}{num_layers[2]}{num_layers[3]}_expa{expand[0]}{expand[1]}{expand[2]}'
    else:
        return

    if cfg.mark is not None:
        cfg.mark = _mark + '_' + cfg.mark
    else:
        cfg.mark = _mark
    return
