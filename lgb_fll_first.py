import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
import lightgbm as lgb
import time
import warnings
warnings.filterwarnings('ignore')

######################################################## data import
df_train = pd.read_csv('训练集/train.csv', engine='python')
df_train.columns = ['time', 'year', 'month', 'day', 'hour', 'minute', 'second',
                    'temp_out', 'humid_out', 'press_out', 'humid_in', 'press_in', 'temp_in']
df_test = pd.read_csv('测试集/test.csv', engine='python')
df_test.columns = ['time', 'year', 'month', 'day', 'hour', 'minute', 'second',
                   'temp_out', 'humid_out', 'press_out', 'humid_in', 'press_in']
df_sample = pd.read_csv('提交样例.csv')

original_feats = ['temp_out', 'humid_out', 'press_out', 'humid_in', 'press_in']
his_feats = ['his_' + x for x in original_feats]

#  标注实际test集中数据
df_test['flag'] = 1

df_train.fillna(method='ffill', inplace=True)

#  对气压显著异常值进行处理
for name in ['press_in', 'press_out']:
    df_train.loc[(df_train[name] < 960) | (df_train[name] > 1000), name] = np.nan
    df_test.loc[(df_test[name] < 960) | (df_test[name] > 1000), name] = np.nan
for name in ['press_in', 'press_out']:
    idx = df_train.loc[df_train[name].isnull().T, name].index
    for i in idx:
        df_train.loc[i, name] = df_train.loc[i - 1:i + 1, name].mean()
    df_test.fillna(method='ffill', inplace=True)

print(df_train.shape)

#  分包含0/12点及不包含0/12点两种情况填充
df_train_without = df_train[~df_train['hour'].isin([0, 12])]
df_test_without = df_test[~df_test['hour'].isin([0, 12])]
fill_temp_out = []
ii = 0
for (fill_train, fill_test) in zip([df_train, df_train_without], [df_test, df_test_without]):
    fill_feats = []
    for df in [fill_train, fill_test]:
        df['humid_io_div'] = np.divide(df['humid_out'], df['humid_in'])
        df['neg_humid_io_div'] = np.divide(100 - df['humid_out'], 100 - df['humid_in'])

        df['press_io_div'] = np.divide(df['press_out'], df['press_in'])
        df['hour_time'] = df['hour'].apply(lambda x: 0 if (x < 8) | (x > 16) else 1)  # 白天还是晚上
        df['humid_press'] = np.add(df['humid_io_div'], df['press_io_div'])

    for name in ['temp_out', 'humid_io_div', 'press_io_div', 'hour_time', 'humid_press']:
        fill_train[name + '1'] = fill_train[name].shift(120)
        fill_train[name + '2'] = fill_train[name].shift(240)
        fill_test[name + '1'] = fill_test[name].shift(1)
        fill_test[name + '2'] = fill_test[name].shift(2)
        fill_feats.append(name + '1')
        fill_feats.append(name + '2')

    for df in [fill_train, fill_test]:
        df['temp_div'] = np.divide(df['temp_out1'], df['temp_out2'])
        df['temp_sub'] = np.subtract(df['temp_out1'], df['temp_out2'])

    fill_feats = fill_feats + ['hour', 'minute', 'temp_div', 'temp_sub']

    df1 = fill_train[fill_feats + ['temp_out']]
    df1 = df1.dropna()
    print(fill_feats)
    trainx = df1[fill_feats]
    trainy = df1['temp_out']
    testx = fill_test[fill_feats]
    testy = fill_test['temp_out']
    print(trainx.shape, testx.shape)

    train_matrix = lgb.Dataset(trainx, label=trainy)
    y = 0
    for seed in [1314, 2333, 90000, 11111, 8888]:
        params = {
                'boosting_type': 'gbdt',
                'max_depth': 5-ii,  # 是否包含0/12点使用不同深度树进行建模
                'objective': 'mse',
                'min_child_weight': 4,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'learning_rate': 0.01,
                'seed': seed,
                'verbose': -1
            }

        model = lgb.train(params, train_matrix, num_boost_round=190+ii*30)
        pred_y = model.predict(testx, ntree_limit=model.best_iteration)
        y += pred_y/5
    this_y = fill_test[['time', 'temp_out']]
    this_y['temp_out'] = y
    fill_temp_out.append(this_y)
    ii = ii+1

aaa = pd.merge(fill_temp_out[0], fill_temp_out[1], how='left', on='time')
ind = aaa[~aaa['temp_out_y'].isnull()].index
#  0/12点之外的采用不包含0/12点数据建模结果
aaa.loc[ind, 'temp_out_x'] = aaa.loc[ind, 'temp_out_y']
df_test['future_temp_out'] = aaa['temp_out_x']

#  获取提交时间
time_list = df_sample.time.map(lambda x: [time.localtime(x).tm_mon, time.localtime(x).tm_mday, time.localtime(x).tm_hour, time.localtime(x).tm_min])
df_sample['month'] = time_list.map(lambda x: x[0])
df_sample['day'] = time_list.map(lambda x: x[1])
df_sample['hour'] = time_list.map(lambda x: x[2])
df_sample['minute'] = time_list.map(lambda x: x[3])

#  加入测试集数据
df_sample = df_sample.merge(df_test[original_feats+['time', 'flag', 'future_temp_out']],
                            on=['time'], how='left')

#  基础填充值获取
#  周期因子法
trend_feats = []
for name in original_feats:
    df_train[name+'_day'] = df_train.groupby(['day']).transform('mean')[name]
    df_train[name + '_trend'] = np.divide(df_train[name], df_train[name+'_day'])
    trend_feats.append(name+'_trend')

df_all = df_train.groupby(['day']).agg('mean')[original_feats]
his_mean_day = df_all.mean()
record = df_train.groupby(['hour', 'minute']).agg('median').reset_index()[['hour', 'minute'] + trend_feats]
record.columns = ['hour', 'minute'] + his_feats
for name, name1 in zip(his_feats, original_feats):
    record.loc[~record['hour'].isin([0, 12]), name] = record.loc[~record['hour'].isin([0, 12]), name].rolling(window=8, center=True, min_periods=1).mean()
    record[name] = record[name] * his_mean_day[name1]

#  直接历史均值法
df_train1 = df_train.drop(index=df_train[df_train['day'].isin([19,20])].index)
record1 = df_train1.groupby(['hour', 'minute']).agg('mean').reset_index()[['hour', 'minute'] + original_feats+['temp_in']]

record1.columns = ['hour', 'minute'] + his_feats+['his_temp_in']
for name in his_feats:
    record1.loc[~record1['hour'].isin([0, 12]), name] = record1.loc[~record1['hour'].isin([0, 12]), name].rolling(window=10, center=True, min_periods=1).mean()

#  两种填充值融合
record[his_feats] = np.add(record[his_feats]*0.35, record1[his_feats]*0.65)
df_sample = df_sample.merge(record, on=['hour', 'minute'], how='left')

#  空缺为12点0/1分，对历史数据bfill不穿越
df_sample[his_feats] = df_sample[his_feats].fillna(method='bfill')


#  利用每2小时数据及历史均值计算基准充填值
tmp = np.subtract(df_sample[original_feats], df_sample[his_feats])
for name in original_feats:
    df_sample.loc[11:, name] = df_sample.index[11:].map(lambda x: np.add(tmp.loc[x-(x-10)%120, name],
                                                                         df_sample.loc[x, 'his_'+name]))

#  室外温度趋势修正
for name in ['temp_out']:
    for i in range(1, 47):
        idx = i * 120 + 10

        #  未来趋势修正
        k1 = (df_sample.loc[idx+119, name]-df_sample.loc[idx, name])/119.  # 填充本期温度斜率
        k2 = (df_sample.loc[idx+120, 'future_'+name]-df_sample.loc[idx, name])/120.  # 预测本期斜率

        #  历史趋势修正
        k11 = (df_sample.loc[idx, name]-df_sample.loc[idx-120, name])/120.  # 实际上期温度斜率
        k21 = (df_sample.loc[idx, 'his_'+name]-df_sample.loc[idx-120, 'his_'+name])/120.  # 历史上期温度斜率

        df_sample.loc[idx:119+idx, name] = df_sample.loc[idx:119+idx, name] + (
                range(120)*(k2-k1)*1. * 0.85 + range(120)*(k11-k21)*(-1.5) * 0.15
        )


#  利用历史均值补充前10个数据
df_sample.loc[:9, original_feats] = df_sample.loc[:9, his_feats].values

df_train_without = df_train[~df_train['hour'].isin([0, 12])]
df_sample_without = df_sample[~df_sample['hour'].isin([0, 12])]
temp_out = []
ii = 0
for (_df_train, _df_sample) in zip([df_train, df_train_without], [df_sample, df_sample_without]):
    print(_df_train.shape, _df_sample.shape)
    feats = original_feats+['hour', 'minute']

    for df in [_df_train, _df_sample]:
        df['hour_time'] = df['hour'].apply(lambda x: 0 if (x < 8) | (x > 16) else 1)
        df['humid_div'] = np.divide(df['humid_in'], df['humid_out'])
        df['humid_div1'] = np.divide(100 - df['humid_in'], 100 - df['humid_out'])
        df['humid_temp_out'] = df['humid_div'] * df['temp_out']
        df['press_div'] = np.divide(df['press_in'] - 900, df['press_out'] - 900)
        df['humid_io'] = df['humid_in'] - df['humid_out']
        df['press_io'] = df['press_in'] - df['press_out']
        df['spe1'] = np.multiply(np.divide(df['humid_out'], df['humid_in']), df['temp_out'])

    feats = feats + ['hour_time', 'humid_div', 'humid_div1', 'humid_temp_out', 'humid_io', 'press_io', 'spe1']
    for df in [_df_train, _df_sample]:
        for name in original_feats:
            df[name + '_halfhour'] = df[name].shift(30)
            df[name + '_onehour'] = df[name].shift(60)
            df[name + '_twohour'] = df[name].shift(120)
            df[name + '_threehour'] = df[name].shift(240)
        #
            df[name + '_halfhour_diff'] = df[name] - df[name + '_halfhour']
            df[name + '_onehour_diff'] = df[name + '_halfhour'] - df[name + '_onehour']
            df[name + '_twohour_diff'] = df[name + '_onehour'] - df[name + '_twohour']
            df[name + '_threehour_diff'] = df[name + '_twohour'] - df[name + '_threehour']
        #
            df[name + '_halfhour_div'] = np.divide(df[name], df[name + '_halfhour'])
            df[name + '_onehour_div'] = np.divide(df[name + '_halfhour'], df[name + '_onehour'])
            df[name + '_twohour_div'] = np.divide(df[name + '_onehour'], df[name + '_twohour'])
            df[name + '_threehour_div'] = np.divide(df[name + '_twohour'], df[name + '_threehour'])

        for name in ['humid_temp_out', 'humid_div', 'humid_div1', 'spe1']:
            df[name + '_halfhour'] = df[name].shift(30)
            df[name + '_onehour'] = df[name].shift(60)
            df[name + '_twohour'] = df[name].shift(120)
            df[name + '_halfhour_diff'] = df[name] - df[name + '_halfhour']
            df[name + '_onehour_diff'] = df[name + '_halfhour'] - df[name + '_onehour']
            df[name + '_twohour_diff'] = df[name + '_onehour'] - df[name + '_twohour']
        #
            df[name + '_halfhour_div'] = np.divide(df[name], df[name + '_halfhour'])
            df[name + '_onehour_div'] = np.divide(df[name + '_halfhour'], df[name + '_onehour'])
            df[name + '_twohour_div'] = np.divide(df[name + '_onehour'], df[name + '_twohour'])
        for name in ['hour_time']:
            df[name + '_halfhour'] = df[name].shift(30)
            df[name + '_onehour'] = df[name].shift(60)
            df[name + '_twohour'] = df[name].shift(120)

    for name in original_feats:
        feats = feats + [name+'_halfhour', name+'_onehour', name+'_twohour']
        feats = feats + [name+'_halfhour_diff', name+'_onehour_diff', name+'_twohour_diff']
        feats = feats + [name+'_halfhour_div', name+'_onehour_div', name+'_twohour_div']

    for name in ['humid_temp_out', 'humid_div', 'humid_div1']:
        feats = feats + [name + '_halfhour', name + '_onehour']
    for name in ['humid_temp_out']:
        feats = feats + [name+'_halfhour_diff', name+'_onehour_diff']
    train_tmp = _df_train[feats+['temp_in']].dropna()
    train_matrix = lgb.Dataset(train_tmp[feats],
                               label=train_tmp['temp_in'] - train_tmp['temp_out'])
    test_matrix = lgb.Dataset(_df_sample[feats])
    ans_y = []
    print(_df_train[feats].shape)
    print(feats)
    pred_valy = 0
    for seed in [2, 2020, 22222, 11111, 8888]:
        params = {
            'boosting_type': 'gbdt',
            'max_depth': 4-ii,
            # 'num_leaves': 14-8*ii,
            'objective': 'mse',
            'min_child_weight': 5,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.5,
            'bagging_freq': 1,
            'learning_rate': 0.01,
            'seed': seed,
            'verbose': -1
        }

        model = lgb.train(params, train_matrix, num_boost_round=2000+ii*100)
        pred_trainy = model.predict(train_tmp[feats], ntree_limit=model.best_iteration)
        mse1 = mse(train_tmp['temp_in'].values, pred_trainy + train_tmp['temp_out'].values)
        print(mse1)

        y = model.predict(_df_sample[feats], ntree_limit=model.best_iteration)
        pred_valy += y/5
    this_y = _df_sample[['time', 'temp_out']]
    this_y['temp_out'] = pred_valy
    temp_out.append(this_y)
    ii = ii+1

aaa = pd.merge(temp_out[0], temp_out[1], how='left', on='time')
print(aaa)
ind = aaa[~aaa['temp_out_y'].isnull()].index
aaa.loc[ind, 'temp_out_x'] = aaa.loc[ind, 'temp_out_y']
df_sample['temperature'] = aaa['temp_out_x'] + df_sample['temp_out']

# test集时刻使用了当前时刻真实值，置为nan后ffill填充
idx = df_sample.loc[df_sample['flag']==1].index
df_sample.loc[idx, 'temperature'] = np.nan
df_sample['temperature'] = df_sample['temperature'].fillna(method='ffill')
print(df_sample['temperature'])
df_sample[['time', 'temperature']].to_csv('submit_ans/lgb_fill_first.csv', header=['time', 'temperature'], index=False)