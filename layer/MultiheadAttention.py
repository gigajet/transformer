import torch
from torch import nn

from ScaledDotProductAttention import ScaledDotProductAttention

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
        self.n_head = n_head
        self.Wq = nn.Linear(d_k_in, d_k)
        self.Wk = nn.Linear(d_k_in, d_k)
        self.Wv = nn.Linear(d_v_in, d_v)
        self.att = ScaledDotProductAttention()
        self.Wo = nn.Linear(d_v, d_v_out)

    """
    Q, K: (*, n, d_k_in)
    V: (*, n, d_v_in)
    mask: (*, n, n) of Boolean, True location is masked, or None if no masking.
    output: (*, n, d_v_out)
    """
    def forward(self, Q, K, V, mask=None):
        Q = self.Wq(Q)  # (*, n, d_k)
        K = self.Wk(K)  # (*, n, d_k)
        V = self.Wv(V)  # (*, n, d_v)
        Q,K,V = self.split(Q), self.split(K), self.split(V)
        an = self.att(Q,K,V,mask) # (*, n_head, n, d_v//n_head)
        an = self.concat(an)
        an = self.Wo(an)
        return an

    """
    A: (*, n, d)
    output: (*, n_head, n, d/n_head)
    """
    def split(self, A):
        n, d = A.shape[-2::]
        A = A.transpose(-2,-1).reshape(*A.shape[:-2], self.n_head, d//self.n_head, n).transpose(-2,-1)
        return A

    """
    A: (*, n_head, n, d//n_head)
    output: (*, n, d)
    """
    def concat(self, A):
        n, ddiv = A.shape[-2::]
        d = ddiv * self.n_head
        A = A.transpose(-2,-1).reshape(*A.shape[:-3], d, n).transpose(-2,-1)
        return A

if __name__=="__main__":
    z = torch.tensor([
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [13,14,15,16],
    ]) # (4, 4)
    n, d = z.shape[-2::]
    z = z.transpose(-2,-1).reshape(*z.shape[:-2], 2, d//2, n).transpose(-2,-1)
    print(z.shape,z) # (2,4,2)

    z = z.transpose(-2,-1).reshape(*z.shape[:-3], 4, 4).transpose(-2,-1)
    print(z.shape,z) # (4,4)

    z = torch.tensor([
        [[1,1,1,1],[2,2,2,2],[3,3,3,3]],
        [[4,4,4,4],[5,5,5,5],[6,6,6,6]],
    ]).float()
    y = torch.ones((2,4,3))
    z = z @ y
    print(z.shape,z)