import torch
from torch import nn

from TransformerEncoderLayer import TransformerEncoderLayer

class TransformerEncoder (nn.Module):
    def __init__(self, n_layer: int,
        d_model: int, d_ff: int, n_head: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, d_model,
            d_model, d_ff, n_head) for _ in range(n_layer)])
    
    """
        x: (*, n, d_model)
        out: (*, n, d_model)
    """
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__=="__main__":
    enc = TransformerEncoder(1, 512, 2048, 8)
    x = torch.randn(3,128,512)
    y = enc(x)
    assert(y.shape == x.shape)

    enc6 = TransformerEncoder(6, 512, 2048, 8)
    y6 = enc(x)
    assert(y.shape == x.shape)
