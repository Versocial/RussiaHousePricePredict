import numpy
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.colors as color
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
def showAsNormDistribution(df,target):
    print(type(df))
    sns.set()
    sns.distplot(df[target], fit=norm)
    fig = plt.figure()
    res = stats.probplot(df[target], plot=plt)
    plt.show()
    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(df[target])
    print('\n miu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

def drawCorrelation(df=pd.read_csv('data/train_f.csv'),target='price_doc'):
    x_cols = [col for col in df.columns if df[col].dtype != 'object' and col!=target]
    labels = []
    values = []
    for col in x_cols:
        labels.append(col)
        v=np.corrcoef(df[col].values, df[target].values)[0, 1]
        values.append(v)
        print(col,v)
    corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})
    corr_df = corr_df.sort_values(by='corr_values')

    ind = np.arange(len(labels))
    width = 0.5
    fig, ax = plt.subplots(figsize=(36, 120))
    rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')

    ax.set_yticks(ind)
    ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
    ax.set_xlabel('Correlation coefficient')
    ax.set_title('Correlation coefficient of the variables')
    plt.show()


def nanData(all_data=pd.read_csv('data/train.csv')):
    ## 找出哪些列有缺失值，以及缺失的比例有多少
    missing_data = all_data.isnull().sum()
    percent = all_data.isnull().sum() / len(all_data)

    missing_col = pd.concat((missing_data, percent), axis=1, keys=['sum', 'percent']).sort_values(by='sum',                                                                                         ascending=False)
    missing_col = missing_col[missing_col['sum'] > 0]['percent']
    plt.figure(figsize=(24,16))
    print(missing_col)
    plt.title('missing data rate')
    missing_col.plot.bar()
    plt.show()

def e():
    print(pd.read_csv('data/train.csv',low_memory=False).describe())

def price_histgram(train_df=pd.read_csv('data/train.csv')):#histgram
    plt.figure(figsize=(8, 6))
    # ax = plt.gca()
    # 用科学记数法
    # ax.ticklabel_format(style='sci', scilimits=(0, 2), axis='y')
    # ax.ticklabel_format(style='sci', scilimits=(0, 1000), axis='x')

    print(train_df.price_doc.values,type(train_df.price_doc.values))
    plt.hist(train_df.price_doc.values, bins=400, density=False, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values))
    plt.xlabel('price', fontsize=12)
    plt.ylabel('count', fontsize=12)
    plt.show()



def drawPriceAreaDiagram(df_train=pd.read_csv('data/train.csv')):
    plt.figure(figsize=(8, 6), dpi=80, num=4)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.scatter(x=df_train['full_sq'], y=df_train['price_doc'])
    plt.xlabel('full_sq（总面积）', fontsize=15)
    plt.ylabel('SalePrice（房价）', fontsize=15)
    plt.show()

    plt.scatter(x=df_train['life_sq'], y=df_train['price_doc'])
    plt.xlabel('life_sq（居住面积）', fontsize=15)
    plt.ylabel('SalePrice（房价）', fontsize=15)
    plt.show()

def showRelation(df_train=pd.read_csv('data/train.csv')):
    print(df_train.select_dtypes(include=object))
    print("ssssssssssssssssskkkkkkkkkkkkkkkkkkkkk")
    # 分布
    showAsNormDistribution(df_train, 'price_doc')
    df_train['price_ln'] = np.log1p(df_train['price_doc'])  # 或者可以尝试 np.log1p(train["SalePrice"]) 表示ln(x+1)
    showAsNormDistribution(df_train, 'price_ln')

    # 相关性矩阵
    corrmat = df_train.corr()
    print(corrmat)
    plt.subplots(figsize=(360, 270))
    sns.heatmap(corrmat, square=True, cmap="RdBu_r")
    plt.show()


def best(df_train=pd.read_csv('data/train_f.csv'),k = 15):
    plt.subplots(figsize=(240,160))
    df_train['price_ln'] = np.log1p(df_train['price_doc'])
    corrmat = df_train.corr()
    cols = corrmat.nlargest(k, 'price_ln')['price_ln'].index
    print(len(df_train.columns))
    cm = np.corrcoef(df_train[cols].values.T)
    sns.heatmap(cm, square=True,cmap="RdBu_r",
                 yticklabels=cols.values, xticklabels=cols.values)
    plt.show()

def showInportance(X_train0=pd.read_csv('data/train_f.csv')):
    from sklearn.linear_model import Lasso
    # X_train=df_train
    y_train=X_train0['price_doc']
    X_train, y_train = (Variable(torch.tensor(np.array(X_train0))), Variable(torch.tensor(np.array(y_train))))
    lasso = Lasso(alpha=0.001)
    lasso.fit(X_train, y_train)
    FI_lasso = pd.DataFrame({"特征重要程度":lasso.coef_}, index=X_train0.columns)
    FI_lasso.sort_values("特征重要程度",ascending=False,inplace=True)
    FI_lasso[FI_lasso["特征重要程度"] != 0].sort_values("特征重要程度").plot(kind="barh", figsize=(15, 25))
    plt.xticks(rotation=90)
    plt.show()


def fixData(df=pd.read_csv('data/train.csv')):

    drawPriceAreaDiagram(df)
    df['temp']=df['price_doc']/df['full_sq']
    df['temp2'] = df['price_doc'] / df['life_sq']
    print(len(df.index))
    varA=df['temp'].min()
    varB=df['temp2'].min()
    df.drop(df[df['temp']==varA].index,inplace=True)
    df.drop(df[df['temp2'] == varB].index, inplace=True)
    print(len(df.index))
    df.drop(columns=['temp','temp2'],inplace=True)
    drawPriceAreaDiagram(df)

    df['life/full']=df['life_sq']/df['full_sq']
    df.drop(columns=['life_sq'],inplace=True)

    df['m/f']=df['male_f']/df['female_f']
    df.drop(columns=['male_f','female_f'],inplace=True)

    df['young_m/f']=df['young_male']/df['young_female']
    df.drop(columns=['young_male','young_female'],inplace=True)

    df['work_m/f'] = df['work_male'] / df['work_female']
    df.drop(columns=['work_male','work_female'],inplace=True)

    df['ekder_m/f'] = df['ekder_male'] / df['ekder_female']
    df.drop(columns=['ekder_male','ekder_female'],inplace=True)

    df['work_rate']=df['work_all']/df['full_all']
    df['young_rate'] = df['young_all'] / df['full_all']
    df['ekder_rate'] = df['ekder_all'] / df['full_all']
    df.drop(columns=['work_all', 'young_all','ekder_all'], inplace=True)

    df.drop(df[df['life/full']>1].index,inplace=True)
    df.drop(df[df['work_rate'] > 1].index, inplace=True)
    df.drop(df[df['young_rate'] > 1].index, inplace=True)
    df.drop(df[df['ekder_rate'] > 1].index, inplace=True)

    df.drop(df[df['max_floor'] <df['floor'] ].index, inplace=True)
    df.drop(df[df['build_year'] > 2020].index,inplace=True)
    df.drop(df[df['build_year'] < 1500].index, inplace=True)


    percent = df.isnull().sum() / len(df)
    print(percent,type(percent))
    df.drop(columns=percent[percent>0.4].index,inplace=True)
    # # df.
    for column in list(df.columns[df.isnull().sum() > 0]):
        mean_val = df[column].mean()
        df[column].fillna(mean_val, inplace=True)

    return df

def showRelation2(df=pd.read_csv('data/train.csv')):
    from pyecharts import options as opts
    from pyecharts.charts import HeatMap
    df['price_ln'] = np.log1p(df['price_doc'])
    xaxis=df.columns.values
    corrmat = df.corr()
    print(corrmat,type(corrmat))
    xaxis=xaxis.tolist()
    print(xaxis,type(xaxis),corrmat.at[xaxis[1],xaxis[2]],type(corrmat.at[xaxis[1],xaxis[2]]),len(xaxis))
    value = [[i, j,corrmat.at[xaxis[i],xaxis[j]] ] for i in range(len(xaxis)) for j in range(len(xaxis))]
    print(value[1])
    # print(type(value),value)
    c = (
        HeatMap()
            .add_xaxis(xaxis)
            .add_yaxis(
            "",
            xaxis,
            value,
            label_opts=opts.LabelOpts(is_show=True, position="inside"),
        )
            .set_global_opts(
            title_opts=opts.TitleOpts(title="相关性热力图"),
            visualmap_opts=opts.VisualMapOpts(),
        )
        #     .render("基础热力图.html")
    )
    c.render_notebook()
    c.render("out/相关性热力图.html")

def afterFixData(df):

    low = df['price_doc'].quantile(0.005)
    high = df['price_doc'].quantile(0.995)
    df.drop(df[(df['price_doc'] < low) | (df['price_doc'] > high)].index, inplace=True)
    price_histgram(df)
    return df

if __name__ == '__main__':
    # 显示所有列
    # pd.set_option('display.max_columns', None)
    # 显示所有行
    # pd.set_option('display.max_rows', None)

    data=pd.read_csv('data/row3.csv')
    data=fixData(data)
    traindf=afterFixData( data[data['isTest']==0]).drop(columns=['isTest'])
    testdf =data[data['isTest'] == 1].drop(columns=['isTest'])
    traindf.to_csv('data/train_f.csv',index=None)
    testdf.to_csv('data/test_f.csv',index=None)

    # traindf=pd.read_csv('data/train_f.csv')
    # drawCorrelation()
    # best(k=30)