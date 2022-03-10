import torch
from math import sqrt
from torch import nn

class DotProductScaledAttention (nn.Module):
    def __init__(self) -> None:
        super().__init__()

    """
    Q, K: (*, n, d_k)
    V: (*, n, d_v)
    mask: (*, n, n) of Boolean, True location is masked, or None if no masking
    """
    def forward (self, Q, K, V, mask=None):
        d_k = K.size(-1)
        score = Q.float() @ K.transpose(-1,-2).float() / sqrt(d_k)
        score = score.softmax(-1)
        if mask is not None:
            # The trailing _ denotes in-place operation
            score.masked_fill_(mask, 0)
        return score @ V.float()

if __name__=="__main__":
    # Unit test goes here
    def test(att,q,k,v,mask,an):
        an_ = att.forward(q.float(),k.float(),v.float(),mask)
        an=an.float()
        # print('model ans', an_)
        # print('correct ans', an)
        assert(an_.allclose(an))
    
    att=DotProductScaledAttention()
    q1=torch.tensor([
        [1,1,1,1],
        [1,1,1,1],
        [1,1,1,1]
    ])
    k1=torch.tensor([
        [1,1,1,1],
        [1,1,1,1],
        [1,1,1,1]
    ])
    v1=torch.tensor([
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1]
    ])
    an1=torch.tensor([
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1]
    ])

    q2=torch.tensor([
        [1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0]
    ])
    k2=torch.tensor([
        [1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0]
    ])
    v2=torch.tensor([
        [1.0,1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0,1.0]
    ])
    an2=torch.tensor([
        [1.0,1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0,1.0]
    ])

    mask1=torch.tensor([
        [0,1,1],
        [0,0,1],
        [0,0,0]
    ]).bool()

    anmask=torch.tensor([
        [1/3,1/3,1/3,1/3,1/3],
        [2/3,2/3,2/3,2/3,2/3],
        [1,1,1,1,1]
    ])

    test(att,q1,k1,v1,None,an1)
    test(att,q2,k2,v2,None,an2)
    test(att,q1,k1,v1,mask1,anmask)
    test(att,q2,k2,v2,mask1,anmask.float())
    print('All sanity check passed')
