from torch import nn

try:
    from layer.MultiheadAttention import MultiheadAttention
    from layer.PositionwiseFeedForward import PositionwiseFeedForward
except:
    from MultiheadAttention import MultiheadAttention
    from PositionwiseFeedForward import PositionwiseFeedForward

class TransformerEncoderLayer (nn.Module):
    def __init__(self, d_in: int, d_out: int,
            d_model: int, d_ff: int, n_head: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiheadAttention(d_in, d_in, d_model, d_model, n_head, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, d_out, dropout)
        self.layer_norm_2 = nn.LayerNorm(d_out)

    """
    x : (*, n, d_in)
    output: (*, n, d_out)
    """
    def forward(self, x):
        x_ = self.dropout(self.self_attention(x,x,x))
        x = self.layer_norm_1(x + x_)
        x_ = self.ffn(x) # FFN have dropout inside
        x = self.layer_norm_2(x + x_)
        return x

if __name__=="__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    dmodel=512; nhead=8
    torch_encoder_layer = nn.TransformerEncoderLayer(dmodel,nhead)
    my_encoder_layer = TransformerEncoderLayer(dmodel, dmodel, dmodel, 2048, nhead)
    # They should be near or exact, as sanity check
    print('Should be near: ',count_parameters(torch_encoder_layer), count_parameters(my_encoder_layer))