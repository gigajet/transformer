import torch
import math
from torch import nn

# See "A Hierarchical Fused FDNN for Data Classification" paper

class FuzzyLayer(nn.Module):

    def __init__(self, output_dim: int, **kwargs):
        super(FuzzyLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.fuzzy_degree = nn.Parameter(torch.empty(output_dim))
        self.sigma = nn.Parameter(torch.ones(output_dim))
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.uniform_(self.fuzzy_degree)

    """
    input: (*, i)
    output: (*, o)
    """
    def forward(self, input, **kwargs):
        y = torch.unsqueeze(input,-1)
        # y (*, i, 1)
        x = torch.repeat_interleave(y, self.output_dim, dim=-1)
        # x (*, i, o) -> (*, o)
        fuzzy_out = torch.exp(
                        -torch.sum(
                            torch.square((x-self.fuzzy_degree)/(self.sigma**2))            
                            ,dim=-2, keepdims=False)
            )
        return fuzzy_out


class FuzzyRuleLayer(nn.Module):

    def __init__(self, input_dim: int, output_dim: int,**kwargs):
        super(FuzzyRuleLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList([
            FuzzyLayer(output_dim) for _ in range(input_dim)
        ])

    """
    input: (*, i)
    output: (*, o)
    """
    def forward(self, input):
        an=torch.ones((*input.size()[:-1], self.output_dim))
        print(an.size())
        for layer in self.layers:
            an=an*layer(input)
        return an

if __name__=="__main__":
    z=FuzzyRuleLayer(5,35)
    y=torch.ones(3,5)
    assert (z(y).shape == torch.Size([3,35]))
    y=torch.ones(3,6,5)
    assert (z(y).shape == torch.Size([3,6,35]))
    print('Sanity check passed')