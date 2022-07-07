import time
import datetime

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
from sklearn.neural_network import MLPRegressor
import numpy as np

from sklearn import metrics
import joblib
from sklearn import preprocessing

features = ['full_sq', 'life_sq', 'max_floor', 'num_room', 'price_doc']

def test(model_mlp,X_train, y_train,name='test'):
    startTime = time.time()
    mlp_score = model_mlp.score(X_train, y_train)
    print('sklearn多层感知器-回归模型得分', mlp_score)  # 预测正确/总数
    result = model_mlp.predict(X_train)
    stopTime = time.time()
    print('总时间是：', stopTime - startTime)
    plt.title(name+"score:%.4f; r2:%.4f; rmse:%.4f"%(mlp_score,
                    metrics.r2_score(y_train,result),metrics.accuracy_score(y_train,result),
                    metrics.mean_squared_error(y_train,result)**0.5))
    plt.plot(y_train, result, 'ro')
    plt.show()


def train():
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)

    x=pd.read_csv('data/train.csv')
    print(x['time'])
    x=x[features]
    print(x.info())
    # print(np.isinf(x).any(),np.isnan(x).any())
    # # print(x['max_floor'])
    # # x=x.apply(lambda x:x.astye(numpy.float))
    # x.applymap(lambda x:x if x!=numpy.nan or x!=numpy.inf  or x!=numpy.NAN or x!=numpy.NaN else 0)
    # print(np.isinf(x).any(),np.isfinite(x).all(),np.isnan(x).any())
    #
    # x.to_csv('data/train0.csv',index=None)

    # print(x)
    x.dropna(axis=0, how='any', inplace=True)
    y=x['price_doc']
    x.drop(columns=['price_doc'],inplace=True)

    print(len(x.index))

    inputNum=len(x.columns)
    X_train,y_train =(Variable(torch.tensor(np.array(x))),Variable(torch.tensor(np.array(y))))

    print(X_train,y_train)

    model_mlp = MLPRegressor(
        hidden_layer_sizes=(25,20,15),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,
        random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_mlp=model_mlp.fit(X_train, y_train)

    test(model_mlp,X_train,y_train,name='train')

    #保存
    joblib.dump(model_mlp, 'out/model.pkl')

if __name__ == '__main__':
        X_test=pd.read_csv('data/test.csv')[features]
        y_test=pd.read_csv('data/test.csv')['price_doc']

        X_test.dropna(axis=0, how='any', inplace=True)
        y_test=X_test['price_doc']
        X_test.drop(columns=['price_doc'],inplace=True)
        X_test=Variable(torch.tensor(np.array(X_test)))
        y_test=Variable(torch.tensor(np.array(y_test)))

        model = joblib.load('out/model.pkl')
        test(model, X_test, y_test)
        print('')