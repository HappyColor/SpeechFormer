
import os
import random
import numpy as np
import torch

def visible_gpus(gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    print('Use GPU:', gpu_id)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def set_backends():
    torch.backends.cudnn.deterministic = True  # True -> Use deterministic algorithms
    torch.backends.cudnn.benchmark = True     # False -> Use deterministic convolution algorithms (slow in GPU)
    print('Use deterministic algorithms')
    print('Use deterministic convolution algorithms (slow in GPU)')

