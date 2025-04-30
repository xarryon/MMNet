import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == 'lrelu':
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Mlp(nn.Module):
    
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)
        self.act_fn = _get_activation_fn("relu")
        self.act_fn2 = nn.Sigmoid()
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
            
    def forward(self, x):
        x = x.transpose(1,0)

        cls = self.fc1(x)
        cls = self.act_fn(cls)
        cls = self.fc2(cls)
        cls = self.act_fn2(cls)
        
        return cls


class SMTEncoderLayer(nn.Module):
    
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.05,
                 activation="relu", normalize_before=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # 256 4 0.1
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.normal_(self.linear1.bias, std=1e-6)
        nn.init.normal_(self.linear2.bias, std=1e-6)
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, src,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

    def forward(self, src,
                pos: Optional[Tensor] = None):
        
        if self.normalize_before:
            return self.forward_pre(src, pos)
    
    
class SMTEncoder(nn.Module):
    
    def __init__(self, encoder_layer, num_layers, hidden_size, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, src,
                pos: Optional[Tensor] = None):
        
        for layer in self.layers:
            src = layer(src, pos) if pos is not None else layer(src)
        
        if self.norm is not None:
            src = self.norm(src)

        return src

    
class MSMTEncoder(nn.Module):
    
    def __init__(self, encoder, mlp):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp
        
    def forward(self, src, pos: Optional[Tensor] = None):
        cls = self.mlp(self.encoder(src, pos))
        cls = cls.squeeze(-1)
        
        return cls


if __name__ == '__main__':
    src = torch.randn(256,3,256)
    pos_embed = torch.randn(256,3,256) 
    
    encoderlay = SMTEncoderLayer(256,4)
    encoder = SMTEncoder(encoderlay, 2, 256, norm=nn.LayerNorm(256))
    mlp = Mlp()
    mencoders = MSMTEncoder(encoder, mlp)
    
    y = mencoders(src, pos_embed)
    print(y.shape)
