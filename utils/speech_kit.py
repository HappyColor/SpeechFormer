
import numpy as np
import librosa
import math
import torch

class Speech_Kit():
    def __init__(self, mode, length=0, feature_dim=0, pad_value=0):
        self.pad_value = pad_value
        self.t2 = length
        self.f2 = feature_dim
        self.mode = mode

    def pad_spec(self, x: np.ndarray):
        '''
        input shape: t, f
        mode: constant or repeat, repeat only performs on time axis
        output shape: t2, f2
        '''

        t, f = x.shape
        x = np.pad(x, ((0,0), (0,self.f2-f)), 'constant', constant_values=(self.pad_value, self.pad_value)) if self.f2>f else x[:, :self.f2]

        if self.t2>t:
            if self.mode=='constant':
                x = np.pad(x, ((0,self.t2-t), (0,0)), 'constant', constant_values=(self.pad_value, self.pad_value))
            elif self.mode=='repeat':
                time = math.ceil(self.t2/t)
                x = np.tile(x, (time, 1))
                x = x[:self.t2]
            else:
                raise ValueError(f'Unknown pad mode:{self.mode}')
        else:
            x = x[:self.t2]

        return torch.from_numpy(x)

def get_D_P(M):
    '''
    calculate db-based spectrogram and power spectrogram from magnitude spectrogram.
    '''
    D = librosa.amplitude_to_db(M, ref=np.max)
    P = M ** 2
    
    return D, P

