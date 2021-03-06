import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
print(x)
y = x.pow(3)+0.1*torch.randn(x.size())
print(y)
x , y =(Variable(x),Variable(y))

# plt.scatter(x,y)
# plt.scatter(x.data,y.data)
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

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
        out = torch.sigmoid(out)
        out =self.predict(out)
        return out

net = Net(1,20,10,1)
print(net)

optimizer = torch.optim.SGD(net.parameters(),lr = 0.1)
loss_func = torch.nn.MSELoss()

plt.ion()
plt.show()

for t in range(5000):
    prediction = net.forward(x)
    loss = loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(t)
    # if t%5 ==0:
    #     plt.cla()
    #     plt.scatter(x.data.numpy(), y.data.numpy())
    #     plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    #     plt.text(0.5, 0, 'Loss = %.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
    #     plt.title(str(t))
    #     plt.pause(0.05)
plt.cla()
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
plt.text(0.5, 0, 'Loss = %.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
plt.title(str(t))
plt.ioff()
plt.title(str(t))
# plt.legend()
plt.show()