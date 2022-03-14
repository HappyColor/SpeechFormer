
import pandas as pd
import numpy as np
import os
import math
import torch
import re
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from scipy import io
from utils.speech_kit import Speech_Kit, get_D_P
import multiprocessing as mp
from sklearn.model_selection import StratifiedShuffleSplit

def identity(x):
    return x

class DistributedDalaloaderWrapper():
    def __init__(self, dataloader: DataLoader, collate_fn):
        self.dataloader = dataloader
        self.collate_fn = collate_fn
    
    def _epoch_iterator(self, it):
        for batch in it:
            yield self.collate_fn(batch)

    def __iter__(self):
        it = iter(self.dataloader)
        return self._epoch_iterator(it)

    def __len__(self):
        return len(self.dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset

    def set_epoch(self, epoch: int):
        self.dataloader.sampler.set_epoch(epoch)

def universal_collater(batch):
    all_data = [[] for _ in range(len(batch[0]))]
    for one_batch in batch:
        for i, (data) in enumerate(one_batch):
            all_data[i].append(data)
    return all_data

class Base_database():
    def __init__(self, names, labels, matdir=None, matkey=None, state=None, label_conveter=None):
        self.names = names
        self.labels = labels
        self.state = state
        self.matdir = matdir
        self.matkey = matkey
        self.conveter = label_conveter
        
    def get_wavfile_label(self, name):
        idx = self.names.index(name)
        label = self.labels[idx]
        return label

    def load_a_sample(self, idx=0):
        label = self.labels[idx]
        x = np.float32(io.loadmat(os.path.join(self.matdir, self.names[idx]))[self.matkey])
        y = torch.tensor(self.label_2_index(label))
        return x, y
    
    def get_sample_name(self, idx):
        return self.names[idx]
    
    def label_2_index(self, label):
        index = self.conveter[label]
        return index

class Base_dataset(Dataset):
    def __init__(self, database: Base_database, mode, length, feature_dim, pad_value, load_name=False):
        super().__init__()
        self.database = database
        self.kit = Speech_Kit(mode, length, feature_dim, pad_value)
        self.load_name = load_name
    
    def __len__(self):
        return len(self.database.names)

    def __getitem__(self, idx):
        return _getitem(idx, self.database, self.kit, self.load_name)

class IEMOCAP(Base_database):
    def __init__(self, matdir=None, matkey=None, state=None, meta_csv_file=None):
        df = pd.read_csv(meta_csv_file)
        df_sad = df[df.label == 'sad']
        df_neu = df[df.label == 'neu']
        df_ang = df[df.label == 'ang']
        df_hap = df[df.label == 'hap']
        df_exc = df[df.label == 'exc']
        df_list = [df_sad, df_neu, df_ang, df_hap, df_exc]
        df = pd.concat(df_list)

        names, labels = [], []
        for row in df.iterrows():
            names.append(row[1]['name'])
            labels.append(row[1]['label'])

        label_conveter = {'ang': 0, 'neu': 1, 'hap': 2, 'exc': 2, 'sad': 3}
        super().__init__(names, labels, matdir, matkey, state, label_conveter)
    
    def foldsplit(self, fold, strategy='5cv'):
        if strategy == '5cv':
            assert 1<=fold<=5, print('leave-one-session-out 5-fold cross validation, but got fold {}'.format(fold))
        elif strategy == '10cv':
            assert 1<=fold<=10, print('leave-one-speaker-out 10-fold cross validation , but got fold {}'.format(fold))
        else:
            raise KeyError('Wrong cross validation setting')

        name_fold, y_fold = [], []
        if strategy == '5cv':
            testSes = 'Ses0{}'.format(6-fold)
            if self.state == 'test':
                for i, name in enumerate(self.names):
                    if testSes in name:
                        name_fold.append(name)
                        y_fold.append(self.labels[i])
            else:
                for i, name in enumerate(self.names):
                    if testSes not in name:
                        name_fold.append(name)
                        y_fold.append(self.labels[i])
        else:
            gender = 'F' if fold%2 == 0 else 'M'
            fold = math.ceil(fold/2)
            testSes = 'Ses0{}'.format(6-fold)
            if self.state == 'test':
                for i, name in enumerate(self.names):
                    if (testSes in name) and (gender in name.split('_')[-1]):
                        name_fold.append(name)
                        y_fold.append(self.labels[i])
            else:
                for i, name in enumerate(self.names):
                    if (testSes not in name) or (gender not in name.split('_')[-1]): 
                        name_fold.append(name)
                        y_fold.append(self.labels[i])
            
        self.names = name_fold
        self.labels = y_fold

class IEMOCAP_dataset(Base_dataset):
    def __init__(self, matdir, matkey, state, meta_csv_file, length=0, feature_dim=0, 
                pad_value=0, mode='constant', fold=1, strategy='5cv', **kwargs):
        database = IEMOCAP(matdir, matkey, state, meta_csv_file)
        database.foldsplit(fold, strategy)
        super().__init__(database, mode, length, feature_dim, pad_value)

class MELD(Base_database):
    def __init__(self, matdir=None, matkey=None, state=None, meta_csv_file=None):
        assert state in ['train', 'dev', 'test'], print(f'Wrong state: {state}')
        self.set_state(state)
        names, labels = self.load_state_data(meta_csv_file)

        if matdir is not None:
            matdir = os.path.join(matdir, state)

        label_conveter = {'neutral': 0, 'anger': 1, 'joy': 2, 'sadness': 3, 'surprise': 4, 'disgust': 5, 'fear': 6}
        super().__init__(names, labels, matdir, matkey, state, label_conveter)

    def set_state(self, state):
        self.state = state
    
    def load_state_data(self, meta_csv_file):
        df = pd.read_csv(meta_csv_file)

        df = df[df.state == self.state]

        names, labels = [], []
        for row in df.iterrows():
            names.append(row[1]['name'])
            labels.append(row[1]['label'])  
        return names, labels

class MELD_dataset(Base_dataset):
    def __init__(self, matdir, matkey, state, meta_csv_file, length=0, feature_dim=0, 
               pad_value=0, mode='constant', **kwargs):
        database = MELD(matdir, matkey, state, meta_csv_file)
        super().__init__(database, mode, length, feature_dim, pad_value)

class Pitt(Base_database):
    def __init__(self, matdir=None, matkey=None, state=None, meta_csv_file=None, fold=1, seed=2021):
        self.set_state(state)
        label_conveter = {'Control': 0, 'Dementia': 1}
        names, labels = self.load_state_data(meta_csv_file)
        # names, labels = self.split_train_test_10fold(names, labels, fold, label_conveter)    # random split 10-fold CV
        names, labels = self.speaker_independent_split_10fold(names, labels, fold, seed)       # speaker independent split 10-fold CV
        super().__init__(names, labels, matdir, matkey, state, label_conveter)

    def set_state(self, state):
        self.state = state

    def split_train_test_10fold(self, names, labels, fold, conveter):
        y = [conveter[l] for l in labels]
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=2021)
        for i, (train_index, test_index) in enumerate(sss.split(y, y)):
            if i + 1 == fold:
                index = train_index if self.state == 'train' else test_index
                names = [names[i] for i in index]
                labels = [labels[i] for i in index]
                break
        return names, labels
    
    def speaker_independent_split_10fold(self, names, labels, fold, seed=2021):
        speaker = [n[:3] for n in names]   # [:3] represent the speaker id 
        speaker = sorted(list(set(speaker)))
        random.seed(seed)        # use the same seed in each fold!!!
        random.shuffle(speaker)
        start_id = math.ceil(len(speaker) / 10) * (fold - 1)
        end_id = math.ceil(len(speaker) / 10) * (fold)
        test_speaker = speaker[start_id:end_id]
        
        pattern = ''
        for sp in test_speaker:
            pattern += f'({sp}.*)|'
        pattern = pattern[:-1]

        train_names, test_names, train_labels, test_labels = [], [], [], []
        for name, label in zip(names, labels):
            result = re.match(pattern, name)
            if result is not None:
                test_names.append(name)
                test_labels.append(label)
            else:
                train_names.append(name)
                train_labels.append(label)
        fold_names = test_names if self.state == 'test' else train_names
        fold_lables = test_labels if self.state == 'test' else train_labels
        
        return fold_names, fold_lables
        
    def load_state_data(self, meta_csv_file):
        df = pd.read_csv(meta_csv_file)
        df = df[df.valid == True]
        
        names, labels = [], []
        for row in df.iterrows():
            names.append(row[1]['name'])
            labels.append(row[1]['label'])  
        return names, labels

    def load_a_sample(self, idx=0):
        label = self.labels[idx]
        x = np.float32(io.loadmat(os.path.join(self.matdir, label, 'cookie', self.names[idx]))[self.matkey])
        y = torch.tensor(self.label_2_index(label))
        return x, y

class Pitt_dataset(Base_dataset):
    def __init__(self, matdir, matkey, state, meta_csv_file, length=0, feature_dim=0, fold=1, seed=2021,
               pad_value=0, mode='constant', **kwargs):
        database = Pitt(matdir, matkey, state, meta_csv_file, fold=fold, seed=seed)
        super().__init__(database, mode, length, feature_dim, pad_value, load_name=True)

class DAIC_WOZ(Base_database):
    def __init__(self, matdir=None, matkey=None, state=None, meta_csv_file=None):
        assert state in ['train', 'test'], print(f'Wrong state: {state}')  # test represents the development set in this database.
        self.set_state(state)
        names, labels = self.load_state_data(meta_csv_file)

        label_conveter = {'not-depressed': 0, 'depressed': 1}
        super().__init__(names, labels, matdir, matkey, state, label_conveter)

    def set_state(self, state):
        if state == 'test':
            state = 'dev'
        self.state = state
    
    def load_state_data(self, meta_csv_file):
        df = pd.read_csv(meta_csv_file)

        df = df[df.state == self.state]

        names, indexes = [], []
        index_2_label = {0: 'not-depressed', 1: 'depressed'}
        for row in df.iterrows():
            names.append(row[1]['name'])
            indexes.append(row[1]['label'])  
        labels = [index_2_label[idx] for idx in indexes]
        
        mat_names = [n[:-10] for n in names]    # (Participant_ID)_(segment)
        
        return mat_names, labels
    
class DAIC_WOZ_dataset(Base_dataset):
    def __init__(self, matdir, matkey, state, meta_csv_file, length=0, feature_dim=0, 
                pad_value=0, mode='constant', **kwargs):
        database = DAIC_WOZ(matdir, matkey, state, meta_csv_file)
        super().__init__(database, mode, length, feature_dim, pad_value, load_name=True)

        # if state == 'train':
            # self.resample_up()
            # self.resample_down()

    def resample_up(self):
        names = self.database.names
        labels = self.database.labels
        num_d = labels.count('depressed')
        num_nd = labels.count('not-depressed')

        r = num_nd // num_d
        s = num_nd % num_d
        index_d = np.arange(num_d)
        random.shuffle(index_d)

        d_index = [index for (index,value) in enumerate(labels) if value == 'depressed']
        d_names = [names[index] for index in d_index]
        d_labels = [labels[index] for index in d_index]

        d_names_s = [d_names[index_d[i]] for i in range(s)]
        d_labels_s = [d_labels[index_d[i]] for i in range(s)]

        d_names = d_names * (r-1)
        d_labels = d_labels * (r-1)

        d_names.extend(d_names_s)
        d_labels.extend(d_labels_s)

        names.extend(d_names)
        labels.extend(d_labels)

        self.database.names = names
        self.database.labels = labels
    
    def resample_down(self):
        names = self.database.names
        labels = self.database.labels
        num_d = labels.count('depressed')
        num_nd = labels.count('not-depressed')

        index_nd = np.arange(num_nd)
        random.shuffle(index_nd)

        nd_index = [index for (index,value) in enumerate(labels) if value == 'not-depressed']
        nd_names = [names[index] for index in nd_index]
        nd_labels = [labels[index] for index in nd_index]

        nd_names = [nd_names[index_nd[i]] for i in range(num_d)]
        nd_labels = [nd_labels[index_nd[i]] for i in range(num_d)]

        d_index = [index for (index,value) in enumerate(labels) if value == 'depressed']
        d_names = [names[index] for index in d_index]
        d_labels = [labels[index] for index in d_index]

        names = d_names + nd_names
        labels = d_labels + nd_labels

        self.database.names = names
        self.database.labels = labels

class DataloaderFactory():
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, state, **kwargs):
        if self.cfg.dataset.database == 'iemocap':
            dataset = IEMOCAP_dataset(
                mode=self.cfg.dataset.padmode, 
                fold=self.cfg.train.current_fold, 
                strategy=self.cfg.train.strategy,
                state=state,
                **kwargs
            )
        elif self.cfg.dataset.database == 'meld':
            dataset = MELD_dataset(
                mode=self.cfg.dataset.padmode, 
                state=state,
                **kwargs
            )
        elif self.cfg.dataset.database == 'pitt':
            dataset = Pitt_dataset(
                mode=self.cfg.dataset.padmode, 
                fold=self.cfg.train.current_fold,
                seed=self.cfg.train.seed,
                state=state,
                **kwargs
            )
        elif self.cfg.dataset.database == 'daic_woz':
            dataset = DAIC_WOZ_dataset(
                mode=self.cfg.dataset.padmode, 
                state=state,
                **kwargs
            )
        else:
            raise KeyError(f'Unsupported database: {self.cfg.dataset.database}')
        
        collate_fn = universal_collater
        sampler = DistributedSampler(dataset, shuffle=state == 'train')
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.cfg.train.batch_size, 
            drop_last=False, 
            num_workers=self.cfg.train.num_workers, 
            collate_fn=identity,
            sampler=sampler, 
            pin_memory=True,
            multiprocessing_context=mp.get_context('fork'), # quicker! Used with multi-process loading (num_workers > 0)
        )

        return DistributedDalaloaderWrapper(dataloader, collate_fn)

def _getitem(idx: int, database: Base_database, kit: Speech_Kit, load_name: bool):
    x, y = database.load_a_sample(idx)
    if database.matkey == 'spec':
        x, _ = get_D_P(x)    # ndarray
        x = x.transpose(1,0)
    x = kit.pad_spec(x)      # ndarray -> Tensor
    
    name = database.get_sample_name(idx) if load_name else None
    
    return x, y, name
    
