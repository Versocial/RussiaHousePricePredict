import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
from sklearn.neural_network import MLPRegressor
import numpy as np

from sklearn import preprocessing
x=pd.read_csv('data/train.csv')
y=x['price_doc']
x.drop(columns=['price_doc'],inplace=True)
print(x.info())

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
print(x.dtypes)
inputNum=len(x.columns)

X_train,y_train =(Variable(torch.tensor(np.array(x))),Variable(torch.tensor(np.array(y))))
print(X_train,y_train)

model_mlp = MLPRegressor(
    hidden_layer_sizes=(25,15),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
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