
"""
Created on Fri Jan 14 16:36:11 CST 2022
@author: lab-chen.weidong
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from module.utils import _get_activation_fn

class Speech_MSA(nn.Module):
    ''' Speech-based Multi-Head Self-Attention (Speech-MSA)
    
    Input dimension order is (batch_size, seq_len, input_dim).
    '''
    def __init__(self, embed_dim, num_heads, local_size, dropout=0., bias=True, overlap=False):
        super(Speech_MSA, self).__init__()
        self.qdim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.local_size = int(local_size)
        self.overlap = overlap    #  overlap = True may have nondeterministic behavior.

        self.project_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=bias)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.scaling = float(self.head_dim) ** -0.5
    
    def get_overlap_segments(self, x: torch.Tensor, window_size: int):
        '''Get overlap segments for local attention.

        Args: 
            x: Input sequence in shape (B, T, C).
            window_size: The needed length of the segment. Must be an odd number.
        '''
        # assert window_size % 2, f'window_size must be an odd number, but get {window_size}.'
        if not window_size % 2:
            window_size += 1     # window_size must be an odd number
        
        b, t, c = x.shape
        pad_len = (window_size - 1) // 2
        x = F.pad(x, (0, 0, pad_len, pad_len), value=0)

        stride = x.stride()
        out_shape = (b, t, window_size, c)
        out_stride = (stride[0], stride[1], stride[1], stride[2])

        return torch.as_strided(x, size=out_shape, stride=out_stride)

    def forward(self, x):
        '''
        Args:
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            attn_mask: mask that prevents attention to certain positions. This is an additive mask
                (i.e. the values will be added to the attention layer).
        Shape:
            Inputs:
            - x: :math:`(B, T, E)` where T is the target sequence length, B is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(B, S)`, ByteTensor, where B is the batch size, S is the source sequence length.
              3-D key_padding_mask with math:`(B, T, S)` is supported now, where T is the target sequence length.
            - attn_mask: :math:`(T, S)` where T is the target sequence length, S is the source sequence length.
        '''
        bsz, tgt_len = x.shape[:2]

        if self.local_size == -1:
            local_size = tgt_len
            global_attn = True
        else:
            local_size = self.local_size
            global_attn = False

        if not self.overlap:
            need_pad = tgt_len % local_size
            if need_pad:
                pad = local_size - need_pad
                x = F.pad(x, (0, 0, 0, pad), mode='constant', value=0)
                tgt_len += pad
        else:
            need_pad = 0

        Q, K, V = self.project_qkv(x).chunk(3, dim=-1)
        Q = Q * self.scaling
        Q = Q.transpose(0, 1).contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        K = K.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        V = V.transpose(0, 1).contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if (self.overlap) and (not global_attn):
            Q = Q.unsqueeze(dim=2)
            K = self.get_overlap_segments(K, window_size=local_size).transpose(-1, -2)
            V = self.get_overlap_segments(V, window_size=local_size)

            attn_output_weights = torch.matmul(Q, K)
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

            attn_output = torch.matmul(attn_output_weights, V).squeeze(dim=2)
        else:
            Q = Q.contiguous().view(-1, local_size, self.head_dim)
            K = K.contiguous().view(-1, local_size, self.head_dim)
            V = V.contiguous().view(-1, local_size, self.head_dim)

            src_len = K.size(1)
            attn_output_weights = torch.bmm(Q, K.transpose(1, 2))

            assert list(attn_output_weights.size()) == [bsz * self.num_heads * tgt_len / local_size, local_size, src_len]

            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

            attn_output = torch.bmm(attn_output_weights, V)

            assert list(attn_output.size()) == [bsz * self.num_heads * tgt_len / local_size, local_size, self.head_dim]
            attn_output = attn_output.view(bsz * self.num_heads, tgt_len, self.head_dim)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim).transpose(0, 1)
        attn_output = self.project_out(attn_output)

        if need_pad:
            attn_output = attn_output[:, :-pad, :]

        return attn_output

class SpeechFormerEncoder(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim=2304, local_size=0, num_heads=8, dropout=0.1, attention_dropout=0.1, activation='relu', overlap=False):
        super().__init__()
        self.dropout = dropout
        self.activation_fn = _get_activation_fn(activation)

        self.attention = Speech_MSA(embed_dim, num_heads, local_size, attention_dropout, overlap=overlap)
        self.attention_layer_norm = nn.LayerNorm(embed_dim)
        
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
    
    def add_position(self, x, position=None, mask=None):
        '''add position information to the input x

        x: B, T, C
        position: T, C
        mask: B, T
        '''
        if position is None:
            return x
        else:
            B, T = x.shape[:2]
            position = position[:T].unsqueeze(dim=0).repeat(B, 1, 1)  # -> B, T, C
            position = position*((1 - mask.unsqueeze(-1).type_as(x))) if mask is not None else position
            return x + position

    def forward(self, x, x_position=None):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x
        x = self.add_position(x, x_position)

        x = self.attention(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.attention_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x

