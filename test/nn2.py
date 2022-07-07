import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

from sklearn.neural_network import MLPRegressor

from sklearn import preprocessing
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
# print(x)
y = x.pow(3)+0.1*torch.randn(x.size())
# print(y)
X_train,y_train =(Variable(x),Variable(y))
print(X_train,y_train)
model_mlp = MLPRegressor(
    hidden_layer_sizes=(6,5),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,
    random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model_mlp.fit(X_train, y_train)

startTime = time.time()
x1 = x.reshape(-1,1)
mlp_score=model_mlp.score(x1,y)
print('sklearn多层感知器-回归模型得分',mlp_score)#预测正确/总数
result = model_mlp.predict(x1)
stopTime = time.time()
sumTime = stopTime - startTime
print('总时间是：', sumTime)
# inp = [[ele] for ele in X_train]
# pre = clf.predict(inp)
# #print(pre)
plt.plot(X_train, y_train, 'bo')
plt.plot(x1, result, 'ro')
plt.show()