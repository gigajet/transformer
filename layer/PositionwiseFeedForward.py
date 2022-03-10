import torch
from torch import nn

class PositionwiseFeedForward (nn.Module):
    def __init__(self, d_in: int, d_ff: int, d_out: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(d_ff, d_out, bias=True)

    """
        x: (*, d_in)
        output: (*, d_out)
    """
    def forward (self, x):
        return self.lin2(self.relu(self.lin1(x)))

