import torch

z = torch.Tensor([
    [1,2,3],
    [4,5,6],
])
"""
[1,2,3]
[4,5,6]
repeat_interleave( ,repeats=3,dim=1)
|1||1||1|  |2||2||2|   |3||3||3|
|4||4||4|  |5||5||5|   |6||6||6|
[1,1,1,2,2,2,3,3,3]
[4,4,4,5,5,5,6,6,6]
repeat_interleave( ,repeats=4,dim=0)
[1,2,3]
[1,2,3]
[1,2,3]
[1,2,3]
[4,5,6]
[4,5,6]
[4,5,6]
[4,5,6]
"""
print(z.shape)
y = torch.repeat_interleave(z, 3, 1)
print(y.shape,y)

"""
a = [n_0, n_1, ..., n_k]
repeat_interleave(a, m, i)
a = [n_0, ..., n_i-1, n_i*m, n_i+1, ..., n_k]



"""