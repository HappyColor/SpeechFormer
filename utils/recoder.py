
import torch
import json

class BaseRecorder():
    def __init__(self, name='No name', dtype=None, device=torch.device('cpu')):
        self.name = name
        self.dtype = dtype
        self.device = device
        self._data = None
        self.count = None

    def reset(self):
        raise NotImplementedError
    
    def record(self):
        raise NotImplementedError

    def to_file(self):
        raise NotImplementedError
            
    def __str__(self):
        fmtstr = '{name} data: {data} blocks:{count}'
        return fmtstr.format(
            name=self.name,
            data=self._data,
            count=self.count
        )
    
    @property
    def data(self):
        return self._data


class TensorRecorder(BaseRecorder):
    def __init__(self, name='No name', dtype=None, device=torch.device('cpu')):
        super().__init__(name, dtype, device)
        self.reset()
    
    def reset(self):
        self._data = torch.tensor([], dtype=self.dtype, device=self.device)
        self.count = torch.tensor(0, dtype=torch.int, device=self.device)

    @torch.no_grad()
    def record(self, new_data):
        self._data = torch.cat((self._data, new_data))
        self.count += 1

    def to_file(self, f):
        torch.save(self._data, f)
    

class StrRecorder(BaseRecorder):
    def __init__(self, name='No name', dtype=None, device=torch.device('cpu')):
        super().__init__(name, dtype, device)
        self.reset()

    def reset(self):
        self._data = []
        self.count = 0
    
    def record(self, new_data: list):
        self._data.extend(new_data)
        self.count += 1

    def to_file(self, f):
        with open(f, 'w') as _f:
            json.dump(self._data, _f, indent=2)
            _f.close()
