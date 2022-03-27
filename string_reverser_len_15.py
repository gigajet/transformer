"""
Custom task 1: reverse bit string of length 15
"""
import sys
import os
import random
import torch
from torch import nn    
from torch import optim as optim
from torch.utils.data import DataLoader
from layer.Transformer import Transformer
from tqdm import tqdm
from priority_queue import PriorityQueue
import operator
from typing import Optional

from layer.PositionwiseFeedForward import PositionwiseFeedForward
from layer.MultiheadAttention import MultiheadAttention
from layer.PositionalEncodedEmbedding import PositionalEncodedEmbedding

class CustomEncoderLayer (nn.Module):
    def __init__(self, d_model: int, d_ff: int, d_out: int) -> None:
        super().__init__()
        self.ffn = PositionwiseFeedForward(d_model, d_ff, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward (self, x):
        _x = self.ffn(x)
        return self.norm(x+_x)

class CustomDecoderLayer (nn.Module):
    def __init__(self, d_model: int, d_ff: int, d_out: int, n_head: int) -> None:
        super().__init__()
        self.context_attention = MultiheadAttention(d_model, d_model, 
            d_model, d_model, n_head, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, d_out)
        self.norm2 = nn.LayerNorm(d_out)

    def forward (self, x, context, context_mask):
        x_ = self.context_attention(x, context, context, context_mask)
        x = self.norm (x + x_)

        x_ = self.ffn(x)
        x = self.norm2 (x + x_)
        return x

# src_vocab is tgt_vocab
class CustomTransformer (nn.Module):
    def __init__(self, 
        max_seq_len: int, num_encoder_layers: int, num_decoder_layers: int,
        d_model: int, n_head: int, d_ff: int,
        vocab_size: int, padding_idx: Optional[int]=None) -> None:
        super().__init__()
        self.embedding = PositionalEncodedEmbedding(max_seq_len, d_model, vocab_size, padding_idx)
        self.encoders = nn.ModuleList([
            CustomEncoderLayer(d_model, d_ff, d_model) for _ in range(num_encoder_layers)
        ])
        self.decoders = nn.ModuleList([
            CustomDecoderLayer(d_model, d_ff, d_model, n_head) for _ in range(num_encoder_layers)
        ])
        self.linear = nn.Linear(d_model, vocab_size)
        self.padding_idx = padding_idx

    """
    input_encoder: (*, m) of Long in [0,vocab_size-1]
    input_decoder: (*, n) of Long in [0,vocab_size-1]
    output: (*, n, vocab_size)
    """

    def forward (self, input_encoder, input_decoder):
        dec_enc_mask = self.make_pad_mask(input_decoder, input_encoder, self.padding_idx)
        input_encoder = self.embedding(input_encoder) # (*, m, d_model)
        input_decoder = self.embedding(input_decoder) # (*, n, d_model)
        context = input_encoder
        for layer in self.encoders:
            context = layer(context)
        output = input_decoder
        for layer in self.decoders:
            output = layer(output, context, dec_enc_mask)
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

# Common alphabet: 0 and 1, and <sos>
"""
There are 2^15=32768 strings of len 15
The model have to learn
"""

LEN = 15
"""
output: list size n of (x,y)
    where x,y is Tensor: (15,) 
"""
def generate_dataset (n: int):
    output = []
    for _ in range(n):
        x = [random.randint(0,1) for _ in range(LEN)]
        y = [2]
        y.extend(list(reversed(x)))
        x,y = torch.tensor(x), torch.tensor(y)
        output.append((x,y))
    return output

"""
    x: Tensor shape (15,)
    output: Tensor shape (16,)

    [Impl detail]
    Trong [0,1], log là hàm nghịch biến
    Maximizing prob <=> Maximize log of prob

    P([y_1, y_2, y_3, ... y_n])
    = P(y_1)*P(y_2 | y_1)*P(y3 | y2, y1)*...*P(y_n | y1, y2, y_n-1)


    We convert prob to log(prob) to avoid vanishing to 0 problem.
"""
def beam_translate (model, x):
    pq = PriorityQueue([(1,torch.tensor([2]))], operator.gt)
    k=0
    while len(pq)>0:
        prob, yt = pq.pop()
        k+=1
        #if k==3: exit(0)
        if yt.size(-1) == len(x) + 1:
            return yt
        pred = model(x,yt).softmax(-1)[-1:, :].squeeze(-2) #(V,)
        print('lg y',prob, yt)
        # pred = torch.log(pred)
        pred = pred.tolist()
        pq.insert(prob*pred[0], torch.cat((yt, torch.tensor([0])), -1))
        pq.insert(prob*pred[1], torch.cat((yt, torch.tensor([1])), -1))
    # return [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

"""
    x: Tensor shape (15,)
    output: Tensor shape (16,)
"""
def greedy_translate(model, x, replay = False):
    y=torch.tensor([2])
    for i in range(15):
        next_elem = model(x,y).argmax(-1) # (1,)
        y = torch.cat((y, next_elem[-1:]), -1)
        if replay:
            print('i',i+1,'model-out',next_elem)
            print('y',y)
    return y

def another_greedy(model, x, replay = False):
    y=torch.tensor([2])
    for i in range(15):
        next_elem = model(x,y).argmax(-1)
        y = torch.cat((torch.tensor([2]), next_elem),-1)
        if replay:
            print('i',i+1,'y',y)
    return y

def evaluate (model, evalset, translate=greedy_translate)->float:
    model = model.eval()
    total=0; num_correct=0
    for x,y in evalset:
        ypred = translate(model, x)
        total += 1
        # print('y ypred',y.shape,ypred.shape,y,ypred)
        if y.equal(ypred):
            num_correct += 1
    model = model.train()
    return num_correct / total

def train (model, criterion, optimizer, trainset, num_epoch: int, 
    batch_size: int):
    model = model.train()
    data_loader = DataLoader(trainset, batch_size, True)
    running_loss = 0.0
    iter = 0
    for epoch in range(num_epoch):
        print('Epoch',epoch+1)
        for x,y in tqdm(data_loader):
            target = y[:,1:]
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                ypred = model(x,y)[:,:-1,:]
            # print('y ypred target', y.shape, ypred.shape, target.shape)
            # Let the first and second dimension be flatten
            ypred = ypred.reshape(-1,ypred.size(-1)).contiguous()
            target = target.reshape(-1).contiguous()
            # DONE: Tính loss đúng chưa? ANS: Đúng rồi.
            loss = criterion(ypred, target)
            loss.backward()
            optimizer.step()
            # Có loss tức là có gradient.

            running_loss += loss.item()
        print('iter loss',iter,running_loss)
        if running_loss < 1e-6:
            print('Convergent! Prematrue terminate epoch',epoch)
            print('Saving checkpoint...')
            torch.save(model.state_dict(), 'task1.pth')    
        running_loss = 0.0
        # print('Training acc: ',evaluate(model, ds))
        print('Saving checkpoint...')
        torch.save(model.state_dict(), 'task1.pth')

if __name__ == "__main__":
    ds = generate_dataset(5000)
    model = CustomTransformer(16,3,3,256,8,512,3)
    if len(sys.argv)<2:
        model.load_state_dict(torch.load('task1.pth'))
        model.eval()
        x = torch.tensor([1,0,0,1,1,0,0,1,1,0,0,1,1,1,1])
        y_ans = torch.tensor([2,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1])
        for i in range(1,LEN+1):
            print(model(x,y_ans[:i]).argmax(-1))
        y = greedy_translate(model, x)
        print('final', y)
        y_pred = model(x,y_ans)
        y_pred2 = model(x,y_ans[:8])
        print(y_pred.argmax(-1))
        print(y_pred2.argmax(-1))
        x = torch.tensor([1,0,0,0,1,0,0,0,1,0,0,0,1,0,0])
        y_ans = torch.tensor([2,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1])
        for i in range(1,LEN+2):
            print(model(x,y_ans[:i]).argmax(-1))
        print('Final: ',y)
    elif sys.argv[1] == "train":
        # I don't know why but https://blog.floydhub.com/the-transformer-in-pytorch/ say it's important
        if os.path.exists('task1.pth'):
            model.load_state_dict(torch.load('task1.pth'))
        else:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        train(model, criterion, optimizer, ds, 1000, 512)
    elif sys.argv[1] == "eval":
        model.load_state_dict(torch.load('task1.pth'))
        evalset = generate_dataset(2000)
        acc = evaluate(model, evalset)
        print('Evaluation accuracy:', acc*100,'%')
    else:
        pass