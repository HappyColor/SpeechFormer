
from yacs.config import CfgNode as CN
_C = CN(new_allowed=True)
_C.train = CN(new_allowed=True)
_C.model = CN(new_allowed=True)
_C.dataset = CN(new_allowed=True)

_C.train.device = 'cuda'
_C.train.num_workers = 8

# Choose the type of the model
_C.model.type = 'SpeechFormer'   # Transformer, SpeechFormer

# Total epochs for training
_C.train.EPOCH = 120
# The size of a mini-batch 
_C.train.batch_size = 32
# Initial learning rate
_C.train.lr = 0.0005
# Set a random seed for reproducition
_C.train.seed = 123
# Select the GPUs used
_C.train.device_id = '0'
# Whether to find a appropriate initial learning rate automatically
_C.train.find_init_lr = False

# Select a database to train a model
_C.dataset.database = 'iemocap'   # iemocap, meld, pitt, daic_woz
# Select a kind of feature to train a model
_C.dataset.feature = 'wav2vec'    # spec, wav2vec, logmel
# Select a padding mode when preparing data a mini-batch 
_C.dataset.padmode = 'constant'   # constant, repeat

# Special mark for current process
_C.mark = None

