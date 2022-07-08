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

features = ['time','full_sq', 'life/full', 'max_floor', 'num_room', 'price_doc','floor',
            'sport_count_5000','trc_count_5000','trc_sqm_5000','sport_objects_raion',
            'office_sqm_5000','market_count_5000','school_education_centers_raion',
            'cafe_count_5000_price_1000','ekder_m/f','healthcare_centers_raion',
            'church_count_5000','office_count_5000','raion_popul','product_type',
            'preschool_education_centers_raion','mosque_count_1500','cafe_count_5000_price_high',
            '7_14_all','children_school','shopping_centers_raion','leisure_count_5000',
            'ekder_rate','school_education_centers_top_20_raion',
            'radiation_raion','bandwidth_sports','labor_force','employment','cpi','salary','time',
            'load_of_teachers_school_per_teacher','turnover_catering_per_cap','big_road1_1line',
            'rent_price_1room_bus','average_life_exp','deposits_value','eurrub','work_rate','gdp_deflator',
            'rent_price_2room_bus','invest_fixed_assets','invest_fixed_capital_per_cap','gdp_annual',
            'ppi','usdrub','micex_cbi_tr','young_rate','sadovoe_km',
            'zd_vokzaly_avto_km','ttk_km','bulvar_ring_km','kremlin_km','nuclear_reactor_km',
            'office_km','university_km','basketball_km','theater_km','detention_facility_km','catering_km',
            'exhibition_km','swim_pool_km','thermal_power_plant_km','museum_km','radiation_km',
            'big_church_km','fitness_km','metro_min_avto','market_shop_km','shopping_centers_km','metro_min_walk',
            'park_km','public_healthcare_km','area_m','ice_rink_km','ts_km','big_road2_km','school_km','oil_chemistry_km',
            'preschool_km','railroad_station_avto_min','public_transport_station_min_walk','green_part_5000',
            'church_synagogue_km','micex_rgbi_tr','provision_nurse','students_state_oneshift','rts',
            'incineration_raion','young_m/f','work_m/f','oil_urals','ID_big_road2','brent','green_zone_part',
            'indust_part'
            ]

def test(model_mlp,X_train, y_train,name='test'):
    startTime = time.time()
    mlp_score = model_mlp.score(X_train, y_train)
    print('sklearn多层感知器-回归模型得分', mlp_score)  # 预测正确/总数
    result = model_mlp.predict(X_train)
    stopTime = time.time()
    print('总时间是：', stopTime - startTime)
    plt.title(name+"score:%.4f; r2:%.4f; rmse:%.4f"%(mlp_score,
                    metrics.r2_score(y_train,result),
                    metrics.mean_squared_error(y_train,result)**0.5))
    plt.plot(y_train, result, 'ro')
    plt.show()


def train(x=pd.read_csv('data/train_f.csv')):
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)

    # print(x['time'])
    # x=x[features]
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

    print('aaaaaaaaaa\n',len(x.index),len(x.columns))

    inputNum=len(x.columns)
    X_train,y_train =(Variable(torch.tensor(np.array(x))),Variable(torch.tensor(np.array(y))))

    model_mlp = MLPRegressor(
        hidden_layer_sizes=(25, 20, 15), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,
        random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_mlp = model_mlp.fit(X_train, y_train)

    test(model_mlp,X_train,y_train,name='train')

    #保存
    joblib.dump(model_mlp, 'out/model.pkl')

if __name__ == '__main__':
        features = ['full_sq', 'sport_count_5000', 'floor','zd_vokzaly_avto_km', 'num_room', 'price_doc']
        # df_train=pd.read_csv('data/train_f.csv')
        x = pd.read_csv('data/train_f.csv')
        c = [a for a in features if a not in x.columns.values]
        print(c)
        train(x=pd.read_csv('data/train_f.csv')[features])
        X_test=pd.read_csv('data/test_f.csv')[features].drop(columns=['price_doc'])
        y_test=pd.read_csv('data/test_f.csv')['price_doc']
        #
        # # X_test.dropna(axis=0, how='any', inplace=True)
        # y_test=X_test['price_doc']
        # X_test.drop(columns=['price_doc'],inplace=True)
        X_test=Variable(torch.tensor(np.array(X_test)))
        y_test=Variable(torch.tensor(np.array(y_test)))
        #
        model = joblib.load('out/model.pkl')
        test(model, X_test, y_test)
        # print('')