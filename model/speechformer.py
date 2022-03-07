import torch
import torch.nn as nn
import torch.nn.functional as F
from module.speechformer_layer import SpeechFormerEncoder
from module.utils import create_PositionalEncoding

def statistical_information(hop=0.01):  # unit: second
    hop *= 1000
    Merge = [50, 250, 1000]
    Locals = [50, 400, 2000]
    Merge = [s//hop for s in Merge]
    Locals = [l//hop for l in Locals]
    Merge.append(-1)    # -1 means global
    Locals.append(-1)
    return Merge, Locals

class MergeBlock(nn.Module):
    ''' Merge features between tow phases.

        The number of tokens is decreased while the dimension of token is increased.
    '''
    def __init__(self, in_channels, merge_scale:int, expand:int=2):
        super().__init__()

        out_channels = in_channels * expand
        self.MS = int(merge_scale)
        self.pool = nn.AdaptiveAvgPool2d((1, in_channels))
        self.fc = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x:torch.Tensor):
        B, T, C = x.shape
        ms = T if self.MS == -1 else self.MS

        need_pad = T % ms
        if need_pad:
            pad = ms - need_pad
            x = F.pad(x, (0, 0, 0, pad), mode='constant', value=0)
            T += pad

        x = x.view(B, T//ms, ms, C)
        x = self.pool(x).squeeze(dim=-2)
        x = self.norm(self.fc(x))

        return x

def make_layers(Merge, Locals, input_dim, ffn_embed_dim, num_layers, num_heads, dropout, attention_dropout, expand, use_position=True):
    layers = []
    last_merge = 1
    for ms, l, exp, num in zip(Merge, Locals, expand, num_layers):
        _l = l // last_merge if l != -1 else -1
        _ms = ms // last_merge if ms != -1 else -1
        module1 = SpeechFormerBlock(num, input_dim, ffn_embed_dim, _l, num_heads, dropout, attention_dropout, use_position=use_position)
        layers += [module1]

        if _ms != -1:
            module2 = MergeBlock(input_dim, _ms, expand=exp)
            layers += [module2]
            input_dim *= exp
            ffn_embed_dim *= exp
        
        last_merge = ms
        if use_position:
            use_position = not use_position   # only the first layer use positional embedding.
    return nn.Sequential(*layers)

class SpeechFormerBlock(nn.Module):
    def __init__(self, num_layers, embed_dim, ffn_embed_dim=2304, local_size=0, num_heads=8, dropout=0.1, attention_dropout=0.1, activation='relu', use_position=False):
        super().__init__()
        self.position = create_PositionalEncoding(embed_dim) if use_position else None
        self.input_norm = nn.LayerNorm(embed_dim)
        self.local = int(local_size)
        self.layers = nn.ModuleList([SpeechFormerEncoder(embed_dim, ffn_embed_dim, local_size, num_heads, dropout, attention_dropout, activation, overlap=True) for _ in range(num_layers)])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        output = self.input_norm(x)

        for layer in self.layers:
            output = layer(output, self.position)

        return output

class SpeechFormer(nn.Module):
    def __init__(self, input_dim, ffn_embed_dim, num_layers, num_heads, hop, num_classes, expand, dropout=0.1, attention_dropout=0.1, device='cuda', **kwargs):
        super().__init__()
        
        self.input_dim = input_dim//num_heads * num_heads
        Merge, Locals = statistical_information(hop)
        assert isinstance(num_layers, list)

        self.layers = make_layers(Merge, Locals, self.input_dim, ffn_embed_dim, num_layers, num_heads, dropout, attention_dropout, expand)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        dim_expand = 1
        expand.pop()
        for e in expand:
            dim_expand *= e

        classifier_dim = self.input_dim * dim_expand
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, classifier_dim//2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(classifier_dim//2, classifier_dim//4),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(classifier_dim//4, num_classes),
        )
    
    def forward(self, x):
        if self.input_dim != x.shape[-1]:
            x = x[:, :, :self.input_dim]

        x = self.layers(x).squeeze(dim=1)
        x = self.avgpool(x.transpose(-1, -2)).squeeze(dim=-1)
        pred = self.classifier(x)
        
        return pred


