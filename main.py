# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd

def joinTables():
    df=pd.read_csv('data/house.csv').set_index('id')
    ecoDf=pd.read_csv('data/eco.csv')
    # test data
    testDf=df.loc[df['timestamp']>='2015/1/1']
    testDf=pd.merge(testDf,ecoDf,on='timestamp',how='left')
    # testDf.to_csv('data/test_row.csv',index=None)
    #train data
    trainDf=df.loc[df['timestamp']<'2015/1/1']
    trainDf=pd.merge(trainDf,ecoDf,on='timestamp',how='left')
    # trainDf.to_csv('data/train_row.csv',index=None)

    trainDf['isTest']=0
    testDf['isTest']=1
    trainDf.append(testDf).to_csv('data/row.csv',index=None)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    joinTables()
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.set()
    # flights_long = sns.load_dataset("flights")
    # flights = flights_long.pivot("month", "year", "passengers")
    # #绘制x-y-z的热力图，比如 年-月-销量 的热力图
    # f, ax = plt.subplots(figsize=(9, 6))
    # #使用不同的颜色
    # sns.heatmap(flights, fmt="d",cmap='YlGnBu', ax=ax)
    # #设置坐标字体方向
    # label_y = ax.get_yticklabels()
    # plt.setp(label_y, rotation=360, horizontalalignment='right')
    # label_x = ax.get_xticklabels()
    # plt.setp(label_x, rotation=45, horizontalalignment='right')
    # plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
