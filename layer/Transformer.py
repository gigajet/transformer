from typing import Optional
import torch
from torch import nn

try:
    from layer.PositionalEncodedEmbedding import PositionalEncodedEmbedding
    from layer.TransformerDecoder import TransformerDecoder
    from layer.TransformerEncoder import TransformerEncoder
except:
    from PositionalEncodedEmbedding import PositionalEncodedEmbedding
    from TransformerDecoder import TransformerDecoder
    from TransformerEncoder import TransformerEncoder

class Transformer (nn.Module):
    def __init__(self, 
        max_seq_len: int, num_encoder_layers: int, num_decoder_layers: int,
        d_model: int, n_head: int, d_ff: int,
        src_vocab_size: int, tgt_vocab_size: int,
        src_padding_idx: Optional[int]=None, tgt_padding_idx: Optional[int]=None) -> None:
        super().__init__()
        self.input_embedding = PositionalEncodedEmbedding(max_seq_len, d_model, src_vocab_size, src_padding_idx)
        self.output_embedding = PositionalEncodedEmbedding(max_seq_len, d_model, tgt_vocab_size, tgt_padding_idx)
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, d_ff, n_head)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, d_ff, n_head)
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self.src_padding_idx = src_padding_idx
        self.tgt_padding_idx = tgt_padding_idx

    """
    input_encoder: (*, m) of Long in [0,src_vocab_size-1]
    input_decoder: (*, n) of Long in [0,tgt_vocab_size-1]
    output: (*, n, tgt_vocab_size)
    """

    def forward (self, input_encoder, input_decoder):
        # NOTE: two embedding weight matrices are currently disjoint, no shared like described in the paper.
        decoder_self_mask = self.make_pad_mask(input_decoder, input_decoder, self.tgt_padding_idx)
        if decoder_self_mask is None:
            decoder_self_mask = torch.triu(torch.ones(input_decoder.size(-1), input_decoder.size(-1)), 1)
        else:
            decoder_self_mask.bitwise_or_(torch.triu(torch.ones(input_decoder.size(-1), input_decoder.size(-1)).bool(), 1))
        dec_enc_mask = self.make_pad_mask(input_decoder, input_encoder, self.src_padding_idx)
        input_encoder = self.input_embedding(input_encoder) # (*, m, d_model)
        input_decoder = self.output_embedding(input_decoder) # (*, n, d_model)
        context = self.encoder(input_encoder)
        output = self.decoder(input_decoder, context, decoder_self_mask, dec_enc_mask)
        output = self.linear(output)
        return output

    """
    row: (*, n)
    col: (*, m)
    pad_idx: int?
    output: (*,1,n,m) of Boolean where a[i,j] True iff col[j]=pad_idx
        or None if pad_idx is None
    """
    @staticmethod
    def make_pad_mask (row, col, pad_idx: Optional[int]=None):
        if pad_idx is None:
            return None
        n, m = row.size(-1), col.size(-1)
        masked = col.eq(pad_idx).unsqueeze(-2).repeat_interleave(n,-2) # (*,n,m)
        return masked.unsqueeze(-3)

if __name__=="__main__":
    # z = torch.tensor([1,0,1,1,0]).unsqueeze(-2)
    # print(z.repeat_interleave(4,-2))

    transformer = Transformer(20,6,6,512,8,2048,6,7)
    inp_enc = torch.tensor([1,2,3,4,5])
    inp_dec = torch.tensor([4,4,3,1])
    output = transformer(inp_enc, inp_dec)
    assert(output.shape == torch.Size([4,7]))

    # One with the mask
    transformer = Transformer(20,6,6,512,8,2048,6,7,5,6)
    inp_enc = torch.tensor([1,2,3,4,5,5])
    inp_dec = torch.tensor([4,6,3,1,6])
    output = transformer(inp_enc, inp_dec)
    assert(output.shape == torch.Size([5,7]))

    transformer = Transformer(20,6,6,512,8,2048,6,7,5,6)
    inp_enc = torch.tensor([[1,2,3,4,5,5],[1,4,0,5,5,5]])
    inp_dec = torch.tensor([[4,6,3,1,6],[4,0,1,6,6]])
    output = transformer(inp_enc, inp_dec)
    assert(output.shape == torch.Size([2,5,7]))

    print('Sanity check passed')
