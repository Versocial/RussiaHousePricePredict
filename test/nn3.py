#coding:utf-8
from sklearn import neural_network
import matplotlib.pyplot as plt
import numpy as np
import sys

mlp = neural_network.MLPRegressor(hidden_layer_sizes=(10,10,10), activation="relu",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001,
                 power_t=0.5, max_iter=200,tol=1e-4)
x = np.arange(-3.0, 3.0, 0.1)
y = np.exp(x)
mlp.fit(np.asarray(x).reshape([-1,1]),y)
inp = [[ele] for ele in x]
pre = mlp.predict(inp)
plt.plot(np.asarray(x), np.asarray(y), 'bo')
plt.plot(np.asarray(inp), np.asarray(pre), 'ro')
plt.show()