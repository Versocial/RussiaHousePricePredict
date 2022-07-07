import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.colors as color
import matplotlib.pyplot as plt

def showAsNormDistribution(df,target):
    sns.set()
    sns.distplot(df[target], fit=norm)
    fig = plt.figure()
    res = stats.probplot(df[target], plot=plt)
    plt.show()
    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(df[target])
    print('\n miu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

def drawCorrelation(df=pd.read_csv('data/train.csv'),target='price_doc'):
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

def a():
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)

    df_train=pd.read_csv('data/train.csv')
    print(df_train.dtypes)
    print(df_train.info())

    # 显示所有列
    pd.set_option('display.max_columns', 10)
    # 显示所有行
    pd.set_option('display.max_rows', 10)
    print("sssssssssssssssss")
    print(df_train.select_dtypes(include=object))
    print("ssssssssssssssssskkkkkkkkkkkkkkkkkkkkk")
    #分布
    showAsNormDistribution(df_train,'price_doc')
    df_train['price_ln'] = np.log1p(df_train['price_doc'])  # 或者可以尝试 np.log1p(train["SalePrice"]) 表示ln(x+1)
    showAsNormDistribution(df_train,'price_ln')
    #相关性矩阵
    corrmat = df_train.corr()
    # print(corrmat)
    plt.subplots(figsize=(360,270))
    sns.heatmap(corrmat,  square=True,cmap="RdBu_r")
    plt.show()

def c():
    df_train=pd.read_csv('data/train.csv')
    # scatterplot
    sns.set()
    cols = ['price_doc', '', 'TotalBsmtSF', 'build']
    sns.pairplot(df_train[cols], size=2.5)
    plt.show();

def d():
    all_data=pd.read_csv('data/train.csv')
    ## 找出哪些列有缺失值，以及缺失的比例有多少
    missing_data = all_data.isnull().sum()
    percent = all_data.isnull().sum() / len(all_data)

    missing_col = pd.concat((missing_data, percent), axis=1, keys=['sum', 'percent']).sort_values(by='sum',                                                                                         ascending=False)
    missing_col = missing_col[missing_col['sum'] > 0]
    print(missing_col)

def e():
    print(pd.read_csv('data/train.csv',low_memory=False).describe())

def p():
    train_df=pd.read_csv('data/train.csv')
    plt.figure(figsize=(8, 6))
    plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values))
    plt.xlabel('index', fontsize=12)
    plt.ylabel('price', fontsize=12)
    plt.show()

def t():
    train_df = pd.read_csv('data/train.csv',low_memory=False)
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
    drawCorrelation()