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
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

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
    timeStdS=time.strptime('2010/1/1',"%Y/%m/%d")
    timeStdE = time.strptime('2020/1/1', "%Y/%m/%d")
    def theT(t):
        return int(time.mktime(t))
    return float(theT(timeArray)-theT(timeStdS))*(2+10*365)/(theT(timeStdE)-theT(timeStdS))


def standardization(df):
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
        "satisfactory": 2, "excellent": 6, "poor": -2, "good": 4, "no data": numpy.nan
    }
    df['ecology']=df['ecology'].map(ecologyMap)
    # print(df['time'])
    return df

def variance_threshold_selector(data, threshold=0.0):
    if threshold==0.0:
        selector = VarianceThreshold()
    else:
        selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

def clean(df):
    df.drop(columns=['timestamp','sub_area'],inplace=True)
    # 方差选择法，返回值为特征选择后的数据
    # 参数threshold为方差的阈值
    df=variance_threshold_selector(df)
    df.dropna(thresh=len(df.columns)*2/3,inplace=True)#删除1/3以上为空的行

def t(train_df = pd.read_csv('data/train.csv',low_memory=False)):
    train_df['yearmonth'] = train_df['timestamp'].apply(lambda x: x[:4] + x[5:7] if x[6]!='/'else x[:4] +'0'+ x[5])

    grouped_df = train_df.groupby('yearmonth').agg({'price_doc':'mean'}).reset_index()
    print(grouped_df,grouped_df.columns)
    plt.plot(grouped_df.yearmonth.values, grouped_df.price_doc.values)
    # sns.barplot( alpha=0.8, color=color[2])
    plt.ylabel('Mean Price', fontsize=12)
    plt.xlabel('Year Month', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()

    grouped_df2 = train_df.groupby('yearmonth').agg({'price_doc': 'median'}).reset_index()
    print(grouped_df2, grouped_df2.columns)
    plt.plot(grouped_df2.yearmonth.values, grouped_df2.price_doc.values)
    # sns.barplot( alpha=0.8, color=color[2])
    plt.ylabel('Median Price', fontsize=12)
    plt.xlabel('Year Month', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()



if __name__ == '__main__':
    # t()

    # trainDf = pd.read_csv('data/train_row.csv',low_memory=False)
    # testDf = pd.read_csv('data/test_row.csv',low_memory=False)

    # standardization(trainDf).to_csv('data/train.csv', index=None)
    # standardization(testDf).to_csv('data/test.csv', index=None)

    # trainDf=pd.read_csv('data/train.csv')
    # testDf = pd.read_csv('data/test.csv')

    # clean(trainDf)
    # clean(testDf)

    # trainDf.to_csv('data/train.csv', index=None)
    # testDf.to_csv('data/test.csv',index=None)
    row=pd.read_csv('data/row.csv')
    standardization(row).to_csv('data/row2.csv',index=None)
    row=pd.read_csv('data/row2.csv')
    clean(row)
    row.to_csv('data/row3.csv',index=None)