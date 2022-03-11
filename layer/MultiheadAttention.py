import torch
from torch import nn

from layer.ScaledDotProductAttention import ScaledDotProductAttention

class MultiheadAttention (nn.Module):
    """
        d_k_in: dimension of query and key input
        d_v_in: dimension of value input
        d_k: total dimension internal query and key
        d_v: total dimension internal value
        n_head: number of heads
        d_v_out: dimension of output
    """
    def __init__(self, d_k_in: int, d_v_in: int,
            d_k: int, d_v: int, n_head: int,
            d_v_out: int) -> None:
        super().__init__()
        assert(d_k % n_head == 0 and d_v % n_head == 0)
        self.Wq = nn.Linear(d_k_in, d_k)
        self.Wk = nn.Linear(d_k_in, d_k)
        self.Wv = nn.Linear(d_v_in, d_v)
        self.att = ScaledDotProductAttention()
        self.Wo = nn.Linear(d_v, d_v_out)

    # TODO phần Linear thì như nhân ma trận lớn được
    # Nhưng phần Attention thì không thể ăn gian (cái d_k)
    # Phải split rồi về sau concat đàng hoàng.
    # Còn phần output thì chỉ nhân một ma trận thôi
    """
    Q, K: (*, n, d_k_in)
    V: (*, n, d_v_in)
    mask: (*, n, n) of Boolean, True location is masked, or None if no masking.
    output: (*, n, d_v_out)
    """
    def forward(self, Q, K, V, mask=None):
        Q.matmul_(self.Wq)  # (*, n, d_k)
        K.matmul_(self.Wk)

if __name__=="__main__":
    pass