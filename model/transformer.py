
import torch
import torch.nn as nn
from module.transformer_layer import build_transformer

class Vanilla_Transformer(nn.Module):
    def __init__(self, input_dim, ffn_embed_dim, num_layers, num_heads, num_classes, dropout=0.1, attention_dropout=0.1, **kwargs):
        super().__init__()

        self.input_dim = input_dim//num_heads * num_heads

        self.Trans = build_transformer(self_attn=True, num_layers=num_layers, embed_dim=self.input_dim, ffn_embed_dim=ffn_embed_dim, 
                                        num_heads=num_heads, dropout=dropout, attention_dropout=attention_dropout)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim//2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.input_dim//2, self.input_dim//4),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.input_dim//4, num_classes),
        )
    
    def forward(self, x: torch.Tensor):
        if self.input_dim != x.shape[-1]:
            x = x[:, :, :self.input_dim]

        x = self.Trans(query=x).transpose(1, 2)
        x = self.avgpool(x).squeeze(dim=-1)
        x = self.classifier(x)
     
        return x

class Official_Vanilla_Transformer(nn.Module):
    def __init__(self, input_dim, ffn_embed_dim, num_layers, num_heads, num_classes, dropout=0.1, **kwargs):
        super().__init__()

        self.input_dim = input_dim//num_heads * num_heads

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=num_heads, dim_feedforward=ffn_embed_dim, dropout=dropout)
        self.Trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(self.input_dim))
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim//2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.input_dim//2, self.input_dim//4),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.input_dim//4, num_classes),
        )
    
    def forward(self, x: torch.Tensor):
        if self.input_dim != x.shape[-1]:
            x = x[:, :, :self.input_dim]

        x = x.transpose(0, 1)   # -> (T, B, C)
        x = self.Trans(src=x)

        x = x.permute(1, 2, 0)
        x = self.avgpool(x).squeeze(dim=-1)
        x = self.classifier(x)
     
        return x

def build_vanilla_transformer(official, **kwargs):
    if official:
        return Official_Vanilla_Transformer(**kwargs)
    else:
        return Vanilla_Transformer(**kwargs)


