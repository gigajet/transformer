from typing import Optional
import torch
from torch import nn

class PositionalEncodedEmbedding (nn.Module):
    def __init__(self, max_seq_len: int, d_embed: int,
        voc_size: int, padding_idx: Optional[int] = None) -> None:
        super().__init__()
        self.input_embedding = nn.Embedding(voc_size, d_embed, padding_idx)
        self.positional_encoding = torch.zeros(max_seq_len, d_embed, requires_grad=False)

        """
        PE[i,2j] = sin(i/(10000**(2j/d_embed)))
        PE[i,2j+1] = cos(i/(10000**(2j/d_embed)))
        """
        row_vec = torch.zeros(d_embed)
        row_vec[::2] = torch.arange(0,d_embed,2) / d_embed
        row_vec[1::2] = torch.arange(0,d_embed,2) / d_embed
        # assert row_vec[2j] = row_vec[2j+1] = 2j / d_embed
        row_vec = 10000 ** row_vec

        col_vec = torch.arange(0, max_seq_len, 1).unsqueeze(-1)
        self.positional_encoding = col_vec / row_vec
        torch.sin_(self.positional_encoding[:,::2])
        torch.cos_(self.positional_encoding[:,1::2])
        assert(self.positional_encoding.shape == torch.Size([max_seq_len, d_embed]))

    """
    x: (*, n) of Long in [0, voc_size-1]
    output: (*, n, d_embed) of Float
    """
    def forward (self, x):
        n = x.size(-1)
        x = self.input_embedding(x)
        return x + self.positional_encoding[:n, :]

if __name__=="__main__":
    """
    We want to create a tensor A of shape (5,10) such that:
    A[i,2j] = i+(2j)
    A[i,2j+1] = i+(2j)
    """
    # b = torch.zeros(10)
    # b[::2] = b[1::2] = torch.arange(0, 10, 2)
    # a = torch.arange(0,5,1).reshape(5,1)
    # print(a,b)
    # print(a / b) # Broadcast
    
    emb = PositionalEncodedEmbedding(63, 512, 1024, 1)
    x = torch.randint(0,1023,(63,))
    y = emb(x)
    assert(y.shape == torch.Size([63, 512]))
    print('Sanity check passed')