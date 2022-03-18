"""
Custom task 1: reverse bit string of length 15
"""
import sys
import random
import torch
from torch import nn
from torch import optim as optim
from layer.Transformer import Transformer
from tqdm.notebook import tqdm

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
def translate(model, x):
    y=torch.tensor([2])
    for _ in range(15):
        next_elem = model(x,y).argmax(-1)[-1:] # (1,)
        y = torch.cat((y,next_elem),dim=-1)
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

def train (model, criterion, optimizer, trainset, num_epoch: int, print_every: int):
    running_loss = 0.0
    iter = 0
    for epoch in range(num_epoch):
        print('Epoch',epoch+1)
        for x, y in tqdm(trainset):

            target = y[1:]
            ypred = model(x,y)[:-1,:]
            # print('y ypred target', y.shape, ypred.shape, target.shape)
            optimizer.zero_grad()

            loss = criterion(ypred, target)
            loss.backward()
            optimizer.step()

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
    if sys.argv[1] == "train":
        # I don't know why but https://blog.floydhub.com/the-transformer-in-pytorch/ say it's important
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        train(model, criterion, optimizer, ds, 10, 100)
    elif sys.argv[1] == "eval":
        model.load_state_dict(torch.load('task1.pth'))
        eval = generate_dataset(50000)
        acc = evaluate(model, eval)
        print('Evaluation accuracy:', acc*100,'%')
    else:
        pass