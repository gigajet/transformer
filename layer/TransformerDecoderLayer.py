import torch
from torch import nn

try:
    from layer.MultiheadAttention import MultiheadAttention
    from layer.PositionwiseFeedForward import PositionwiseFeedForward
except:
    from MultiheadAttention import MultiheadAttention
    from PositionwiseFeedForward import PositionwiseFeedForward

class TransformerDecoderLayer (nn.Module):
    def __init__(self, d_in: int, d_model: int, d_ff: int, d_out: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.masked_self_attention = MultiheadAttention(d_in, d_in, 
            d_model, d_model, n_head, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.context_attention = MultiheadAttention(d_model, d_model, 
            d_model, d_model, n_head, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, d_out)
        self.norm3 = nn.LayerNorm(d_out)
        
    """
    x: (*, n, d_in)
    context: (*, m, d_model)
    self_mask: (*, n, n)
    context_mask: (*, n, m)
    output: (*, n, d_out)
    """
    def forward (self, x, context, self_mask, context_mask):
        x_ = self.dropout1(self.masked_self_attention(x, x, x, self_mask))
        x = self.norm1 (x + x_)
        
        x_ = self.dropout2(self.context_attention(x, context, context, context_mask))
        x = self.norm2 (x + x_)

        x_ = self.ffn(x)
        x = self.norm3 (x + x_)
        return x

if __name__=="__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    torch_decoder_layer = nn.TransformerDecoderLayer(512,8)
    d_model = 512
    my_decoder_layer = TransformerDecoderLayer(d_model, d_model, 2048, d_model, 8, 0.1)
    # They should be near or exact, as sanity check
    print(count_parameters(torch_decoder_layer), count_parameters(my_decoder_layer))
    
    # Some sanity check out shape
    dec = TransformerDecoderLayer(512,512,2048,512,8,0.0)
    x = torch.randn(3, 128, 512)
    ctx = torch.randn(3, 128, 512)
    y = dec(x, ctx, None, None)
    assert(x.shape == y.shape)
