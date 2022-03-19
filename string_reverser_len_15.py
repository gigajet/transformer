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
    x: Tensor shape (b,15) or (15,)
    output: Tensor shape (b,16) or (16,)
"""
def translate(model, x, replay = False):
    y=torch.tensor([2])
    for i in range(15):
        next_elem = model(x,y).argmax(-1) # (1,)
        last_next_elem = next_elem[-1:]
        y = torch.cat((y,last_next_elem),dim=-1)
        if replay:
            print(i+1,next_elem,y)
    return y

def evaluate (model, evalset)->float:
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
    batch_size: int, print_every: int):
    trainset = DataLoader(trainset, batch_size, True)
    running_loss = 0.0
    iter = 0
    for epoch in range(num_epoch):
        print('Epoch',epoch+1)
        for x,y in tqdm(trainset):
            target = y[:,1:]
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                ypred = model(x,y)[:,:-1,:]
            # print('y ypred target', y.shape, ypred.shape, target.shape)
            # DONE: Tính loss đúng chưa? ANS: Đúng rồi.
            loss = criterion(ypred, target)
            loss.backward()
            optimizer.step()
            # Có loss tức là có gradient.

            running_loss += loss.item()
            iter += 1
            if iter % print_every == 0:
                evaluate(model, trainset)
                print('iter loss',iter,running_loss / print_every)
                running_loss = 0.0
        print('Training acc: ',evaluate(model, ds))
        print('Saving checkpoint...')
        torch.save(model.state_dict(), 'task1.pth')

if __name__ == "__main__":
    ds = generate_dataset(500)
    model = Transformer(16,3,3,256,4,512,3,3)
    if len(sys.argv)<2:
        model.load_state_dict(torch.load('task1.pth'))
        x = torch.tensor([1,0,0,1,1,0,0,1,1,0,0,1,1,1,1])
        y = translate(model, x, replay=True)
        print('Final: ',y)
    elif sys.argv[1] == "train":
        # I don't know why but https://blog.floydhub.com/the-transformer-in-pytorch/ say it's important
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        train(model, criterion, optimizer, ds, 10, 64, 100)
    elif sys.argv[1] == "eval":
        model.load_state_dict(torch.load('task1.pth'))
        eval = generate_dataset(50000)
        acc = evaluate(model, eval)
        print('Evaluation accuracy:', acc*100,'%')
    else:
        pass