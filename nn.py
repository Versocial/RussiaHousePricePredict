import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_hidden2,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden2)
        self.predict = nn.Linear(n_hidden2,n_output)
    def forward(self,input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out =self.predict(out)
        return out

x=pd.read_csv('data/train.csv')
y=x['price_doc']
x.drop('price_doc')
inputNum=len(x.columns)
net = Net(inputNum,25,15,1)
print(net)

optimizer = torch.optim.SGD(net.parameters(),lr = 0.1)
loss_func = torch.nn.MSELoss()

for t in range(40):
    prediction = net(x)
    loss = loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(t)


torch.save(net.state_dict(), 'net_params'+str(inputNum)+'-25-15-1.pkl')
