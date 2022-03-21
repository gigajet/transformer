"""
Custom task 1: reverse bit string of length 15
"""
import sys
import random
import torch
from torch import nn
from torch import optim as optim
from torch.utils.data import DataLoader
from layer.Transformer import Transformer
from tqdm import tqdm
from priority_queue import PriorityQueue
import operator
import copy
import math

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

    We convert prob to log(prob) to avoid vanishing to 0 problem.
"""
def beam_translate (model, x):
    pq = PriorityQueue([(1,torch.tensor([2]))], operator.gt)
    k=0
    while len(pq)>0:
        prob, yt = pq.pop()
        k+=1
        #if k==3: exit(0)
        if len(yt) == len(x)+1:
            return yt
        pred = model(x,yt).softmax(-1)[-1:, :].squeeze(-2) #(V,)
        print('lg y pred',prob, yt,pred)
        # pred = torch.log(pred)
        pred = pred.tolist()
        pq.insert(prob*pred[0], torch.cat((yt, torch.tensor([0])), -1))
        pq.insert(prob*pred[1], torch.cat((yt, torch.tensor([1])), -1))
    return [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

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
        running_loss = 0.0
        # print('Training acc: ',evaluate(model, ds))
        print('Saving checkpoint...')
        torch.save(model.state_dict(), 'task1.pth')

if __name__ == "__main__":
    ds = generate_dataset(5000)
    model = Transformer(16,1,1,256,8,512,3,3)
    if len(sys.argv)<2:
        model.load_state_dict(torch.load('task1.pth'))
        model.eval()
        x = torch.tensor([1,0,0,1,1,0,0,1,1,0,0,1,1,1,1])
        y_ans = torch.tensor([2,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1])
        # for i in range(1,17):
        #     print(model(x,y_ans[:i]).argmax(-1)[:15])
        # TODO: IMPLEMENT BEAM SEARCH FOR MOST PROBABLE SENTENCE
        y_pred = model(x,y_ans)
        y_pred2 = model(x,y_ans[:8])
        print(y_pred.argmax(-1))
        print(y_pred2.argmax(-1))
        x = torch.tensor([1,0,0,0,1,0,0,0,1,0,0,0,1,0,0])
        y_ans = torch.tensor([2,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1])
        for i in range(1,17):
            print(model(x,y_ans[:i]).argmax(-1)[:15])
        y = beam_translate(model, x)
        print('Final: ',y)
    elif sys.argv[1] == "train":
        # I don't know why but https://blog.floydhub.com/the-transformer-in-pytorch/ say it's important
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        train(model, criterion, optimizer, ds, 40, 256)
    elif sys.argv[1] == "eval":
        model.load_state_dict(torch.load('task1.pth'))
        eval = generate_dataset(50000)
        acc = evaluate(model, eval)
        print('Evaluation accuracy:', acc*100,'%')
    else:
        pass