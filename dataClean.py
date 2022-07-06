import numpy
import numpy as np
import pandas as pd
from datetime import datetime, date
from operator import le, eq
import gc
from sklearn import model_selection, preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import time


def replace(x):
    if x=='yes':
        return 1
    elif x=='no':
        return 0
    elif x=='#!':
        return numpy.nan
    else:
        return x

def timeMap(x):
    timeArray = time.strptime(x, "%Y/%m/%d")
    return int(time.mktime(timeArray))


def clean(df):
    df=df.applymap(lambda  x:replace(x))
    df.loc[df['product_type']=='Investment','product_type']=1
    df.loc[df['product_type']=='OwnerOccupier','product_type']=0
    df['time']=df['timestamp'].map(timeMap)

    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    print(df.dtypes)
    df['old_education_build_share']=df['old_education_build_share'].map(lambda x:str(x).replace(',',''))
    df['modern_education_share'] = df['modern_education_share'].map(lambda x: str(x).replace(',', ''))
    df['child_on_acc_pre_school']=df['child_on_acc_pre_school'].map(lambda x:str(x).replace(',',''))

    ecologyMap = {
        "satisfactory": 2, "excellent": 6, "poor": -2, "good": 4, "no data": 0
    }
    df['ecology']=df['ecology'].map(ecologyMap)
    # print(df['time'])
    return df

if __name__ == '__main__':
    trainDf = pd.read_csv('data/train_row.csv',low_memory=False)
    testDf = pd.read_csv('data/test_row.csv',low_memory=False)
    clean(trainDf).to_csv('data/train.csv',index=None)
    clean(testDf).to_csv('data/test.csv',index=None)