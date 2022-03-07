"""
Created on Tue Sep 28 16:53:37 CST 2021
@author: lab-chen.weidong
"""

import torch
from torch import distributed as dist

def load_model(model_type, device, **kwargs):
    if model_type=='Transformer':
        from model.transformer import build_vanilla_transformer
        model = build_vanilla_transformer(**kwargs)
    elif model_type=='SpeechFormer':
        from model.speechformer import SpeechFormer
        model = SpeechFormer(**kwargs)
    else:
        raise KeyError(f'Unknown model type: {model_type}')

    if device == 'cuda':
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])

    return model


