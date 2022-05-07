import torch
import math
from torch import nn

# See "A Hierarchical Fused FDNN for Data Classification" paper
# See "Support-Vector-Based Fuzzy Neural Network for Pattern Classification" paper

class MembershipFunctionLayer(nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.muy = nn.Parameter(torch.empty(input_dim, output_dim))
        self.sigma = nn.Parameter(torch.ones(input_dim, output_dim))
        nn.init.uniform_(self.muy)

    """
    input: (*, i)
    output: (*, i, o)
    """
    def forward(self, input, **kwargs):
        input = torch.unsqueeze(input,-1) # (*, i, 1)
        temp = -torch.square(input - self.muy) / torch.square(self.sigma) # (*, i, o)
        output = torch.exp(temp)
        return output


class FuzzyRuleLayer(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    """
    input: (*, i, o)
    output: (*, o)
    """
    def forward(self, input):
        output = input.prod(-2)
        return output

if __name__=="__main__":
    """
    z=FuzzyRuleLayer(5,35)
    y=torch.ones(3,5)
    assert (z(y).shape == torch.Size([3,35]))
    y=torch.ones(3,6,5)
    assert (z(y).shape == torch.Size([3,6,35]))
    """
    print('Sanity check passed')