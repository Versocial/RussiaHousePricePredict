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
    testDf.to_csv('data/test_row.csv',index=None)
    #train data
    trainDf=df.loc[df['timestamp']<'2015/1/1']
    trainDf=pd.merge(trainDf,ecoDf,on='timestamp',how='left')
    trainDf.to_csv('data/train_row.csv',index=None)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    joinTables()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
