import torch
from torch import nn
import torch.nn.functional as F

from tools.utils import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer_block import build_transformer

class MMNet(nn.Module):
    def __init__(self, backbone, transformer, hidden_dim, vocab_size, imgsize):
        super().__init__()
        self.backbone = backbone
        self.imgsize = imgsize

        self.transformer = transformer
        self.mlp1 = MLP(hidden_dim, 512, vocab_size, 3)
        self.mlp2 = MLP(hidden_dim, 512, vocab_size, 3)
        self.mlp3 = MLP(hidden_dim, 512, vocab_size, 3)
        
        
    def forward(self, samples, target, target_mask):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(self.imgsize, samples)

        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        h_w = torch.tensor([self.imgsize, self.imgsize]).repeat(src.shape[0], 1).to(src.device)
        h_w = h_w.unsqueeze(0)
        
        hs, src_c, spa = self.transformer(src, mask, pos[-1], target, target_mask, h_w)

        out1 = self.mlp1(hs[0].permute(1, 0, 2))
        out2 = self.mlp2(hs[1].permute(1, 0, 2))
        out3 = self.mlp3(hs[2].permute(1, 0, 2))
    
        return [out1, out2, out3], src_c, spa


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(config):
    backbone = build_backbone(config)
    transformer = build_transformer(config)

    model = MMNet(backbone, transformer, config.hidden_dim, config.vocab_size, config.imgsize)

    return model
