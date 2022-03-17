import torch
from torch import nn

from TransformerDecoderLayer import TransformerDecoderLayer

class TransformerDecoder (nn.Module):
    # Assumption: d_in = d_out = d_model for all attention block
    def __init__(self, n_layer: int,
        d_model: int, d_ff: int, n_head: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, d_model, d_ff, d_model, n_head)
                for _ in range(n_layer)
        ])

    """
    x: (*, n, d_model)
    context: (*, m, d_model_ctx)
    self_mask: (*, n, n) of Boolean. True positions are masked out.
    context_mask: (*, n, m) of Boolean. True positions are masked out.
    output: (*, n, d_model)
    """
    def forward (self, x, context, self_mask, context_mask):
        for layer in self.layers:
            x = layer(x, context, self_mask, context_mask)
        return x

if __name__=="__main__":
    dec = TransformerDecoder(1,512,2048,8)
    x = torch.randn(3, 128, 512)
    ctx = torch.randn(3, 128, 512)
    y = dec(x, ctx, None, None)
    assert(y.shape == x.shape)

    dec6 = TransformerDecoder(6,512,2048,8)
    x = torch.randn(3, 256, 512)
    ctx = torch.randn(3, 256, 512)
    self_mask = torch.triu(torch.ones(256,256)).bool()
    context_mask = torch.zeros(256,256).bool()
    y = dec6(x, ctx, self_mask, context_mask)
    assert(y.shape == x.shape)
    print('All sanity check passed')
