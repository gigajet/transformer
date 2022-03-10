import torch
from torch import nn

class MultiheadAttention (nn.Module):
    def __init__(self, d_k: int, d_v: int, n_head: int) -> None:
        super().__init__()
        assert(d_k % n_head == 0 and d_v % n_head == 0)
        
