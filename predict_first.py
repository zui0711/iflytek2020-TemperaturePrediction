import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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

df_test['flag'] = 1

df_train.fillna(method='ffill', inplace=True)
for name in ['press_in', 'press_out']:
    df_train.loc[(df_train[name] < 960) | (df_train[name] > 1000), name] = np.nan
    df_test.loc[(df_test[name] < 960) | (df_test[name] > 1000), name] = np.nan
for name in ['press_in', 'press_out']:
    idx = df_train.loc[df_train[name].isnull().T, name].index
    for i in idx:
        df_train.loc[i, name] = df_train.loc[i - 1:i + 1, name].mean()
    df_test.fillna(method='ffill', inplace=True)
# df_train.dropna(inplace=True)

print(df_train.shape)

####################################################################
# lgb

feats = original_feats + ['hour', 'minute']

for df in [df_train, df_test]:
    df['hour_time'] = df['hour'].apply(lambda x: 0 if (x < 8) | (x > 16) else 1)
    df['humid_div'] = np.divide(df['humid_in'], df['humid_out'])
    df['humid_div1'] = np.divide(100 - df['humid_in'], 100 - df['humid_out'])
    df['humid_temp_out'] = df['humid_div'] * df['temp_out']
    df['temp_out1'] = df['temp_out'].apply(lambda x: 0 if x < 12.3 else 1)
    df['press_div'] = np.divide(df['press_in'] - 900, df['press_out'] - 900)
    df['humid_io'] = df['humid_in'] - df['humid_out']
    df['press_io'] = df['press_in'] - df['press_out']
    df['spe1'] = np.multiply(np.divide(df['humid_out'], df['humid_in']), df['temp_out'])

feats = feats + ['hour_time', 'humid_div', 'humid_div1', 'humid_temp_out', 'humid_io', 'press_io',
                 'spe1', 'press_div']

for df in [df_train]:
    for name in original_feats + ['spe1', 'humid_temp_out', 'press_div', 'hour_time']:
        df[name + '_twohour'] = df[name].shift(120)
        df[name + '_twohour_diff'] = df[name] - df[name + '_twohour']
        df[name + '_twohour_div'] = np.divide(df[name], df[name + '_twohour'])
for df in [df_test]:
    for name in original_feats + ['spe1', 'humid_temp_out', 'press_div', 'hour_time']:
        df[name + '_twohour'] = df[name].shift(1)
        df[name + '_twohour_diff'] = df[name] - df[name + '_twohour']
        df[name + '_twohour_div'] = np.divide(df[name], df[name + '_twohour'])

for name in original_feats:
    feats = feats + [name + '_twohour']

for name in ['temp_out']:
    feats = feats + [name + '_twohour_diff']
    feats = feats + [name + '_twohour_div']

train_tmp = df_train[feats+['temp_in']].dropna()
train_matrix = lgb.Dataset(train_tmp[feats],
                           label=train_tmp['temp_in'] - train_tmp['temp_out'])
test_matrix = lgb.Dataset(df_test[feats])
ans_y = []
print(df_train[feats].shape)
print(feats)
pred_valy = 0
for seed in [2, 2020, 22222, 11111, 8888]:
    params = {
        'boosting_type': 'gbdt',
        'max_depth': 5,
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

    model = lgb.train(params, train_matrix, num_boost_round=2200)
    pred_trainy = model.predict(train_tmp[feats], ntree_limit=model.best_iteration)
    mse1 = mse(train_tmp['temp_in'].values, pred_trainy + train_tmp['temp_out'].values)
    print(mse1)

    y = model.predict(df_test[feats], ntree_limit=model.best_iteration)
    pred_valy += y/5
df_test['temp_in'] = pred_valy + df_test['temp_out']

####################################################################
#  lr
feats = original_feats + ['hour', 'minute'] + ['hour_time', 'humid_div', 'humid_div1', 'humid_temp_out', 'humid_io', 'press_io',
                     'spe1', 'press_div']
train_tmp = df_train[feats+['temp_in']].dropna()
print(train_tmp.shape)
trainx = train_tmp[feats]
trainy = train_tmp['temp_in']

scalerx = StandardScaler()
trainx = scalerx.fit_transform(trainx)
testx = scalerx.transform(df_test[feats])

model = Ridge(alpha=0.001)
model.fit(trainx, trainy)
pred_trainy = model.predict(trainx)
print(mse(pred_trainy, trainy))

df_test['temp_in'] = df_test['temp_in']*0.6 + model.predict(testx)*0.4

####################################################################
#  lgb预测未来2小时值

df_train_without = df_train[~df_train['hour'].isin([0, 12])]
df_test_without = df_test[~df_test['hour'].isin([0, 12])]
future_temp_in = []
ii = 0
for (fill_train, fill_test) in zip([df_train, df_train_without], [df_test, df_test_without]):
    feats = ['hour', 'minute', 'hour_time']

    for name in ['temp_out', 'temp_in', 'hour_time', 'humid_temp_out', 'spe1']:
        fill_train[name+'1'] = fill_train[name].shift(120)
        fill_train[name+'2'] = fill_train[name].shift(240)
        fill_test[name+'1'] = fill_test[name].shift(1)
        fill_test[name+'2'] = fill_test[name].shift(2)
        feats.append(name+'1')
        feats.append(name+'2')
    for name in ['humid_div', 'humid_div1']:
        fill_train[name + '1'] = fill_train[name].shift(120)
        fill_train[name + '2'] = fill_train[name].shift(240)
        fill_test[name + '1'] = fill_test[name].shift(1)
        fill_test[name + '2'] = fill_test[name].shift(2)
        feats.append(name + '1')
        # feats.append(name + '2')
    for df in [fill_train, fill_test]:
        df['temp_div'] = np.divide(df['temp_out1'], df['temp_out2'])
        df['temp_sub'] = np.subtract(df['temp_out1'], df['temp_out2'])
        df['temp_in_div'] = np.divide(df['temp_in1'], df['temp_in2'])
        df['temp_in_sub'] = np.subtract(df['temp_in1'], df['temp_in2'])
        df['humid_temp_div'] = np.divide(df['humid_temp_out1'], df['humid_temp_out2'])
        df['humid_temp_sub'] = np.divide(df['humid_temp_out1'], df['humid_temp_out2'])
        df['spe1_div'] = np.divide(df['spe11'], df['spe12'])
        df['spe1_sub'] = np.divide(df['spe11'], df['spe12'])
    feats = feats + ['temp_div', 'temp_sub', 'temp_in_div', 'temp_in_sub',
                     'humid_temp_div', 'humid_temp_sub', 'spe1_div', 'spe1_sub']

    train_tmp = fill_train[feats+['temp_in']].dropna()
    train_tmp[feats] = train_tmp[feats]
    train_tmp = train_tmp[feats+['temp_in']].dropna()

    trainx = train_tmp[feats]
    trainy = train_tmp['temp_in']

    test_tmp = fill_test[feats+['temp_in']]
    test_tmp[feats] = fill_test[feats]
    testx = test_tmp[feats].astype('float')
    testy = fill_test['temp_in']

    train_matrix = lgb.Dataset(trainx, label=trainy)
    test_matrix = lgb.Dataset(testx, label=testy)
    print(trainx.shape)
    print(feats)
    pred_valy = 0
    for seed in [2, 2020, 22222, 11111, 8888]:
        params = {
            'boosting_type': 'gbdt',
            'max_depth': 5-ii,
            'objective': 'mse',
            'min_child_weight': 5,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'learning_rate': 0.01,
            'seed': seed,
            'verbose': -1
        }

        model = lgb.train(params, train_matrix, num_boost_round=210+ii*35)
        y = model.predict(testx, ntree_limit=model.best_iteration)
        pred_valy += y/5
    this_y = fill_test[['time', 'temp_in']]
    this_y['temp_in'] = y
    future_temp_in.append(this_y)
    ii += 1
aaa = pd.merge(future_temp_in[0], future_temp_in[1], how='left', on='time')
ind = aaa[~aaa['temp_in_y'].isnull()].index
aaa.loc[ind, 'temp_in_x'] = (aaa.loc[ind, 'temp_in_x']*0.3 + aaa.loc[ind, 'temp_in_y']*0.7)
df_test['future_temp_in'] = aaa['temp_in_x']

time_list = df_sample.time.map(lambda x: [time.localtime(x).tm_mon, time.localtime(x).tm_mday, time.localtime(x).tm_hour, time.localtime(x).tm_min])
df_sample['month'] = time_list.map(lambda x: x[0])
df_sample['day'] = time_list.map(lambda x: x[1])
df_sample['hour'] = time_list.map(lambda x: x[2])
df_sample['minute'] = time_list.map(lambda x: x[3])
df_sample = df_sample.merge(df_test[['time', 'flag', 'temp_in', 'future_temp_in']], on=['time'], how='left')
print(df_sample.columns)

# 加入训练集历史均值
# 周期因子
trend_feats = []
for name in ['temp_in']:
    df_train[name+'_day'] = df_train.groupby(['day']).transform('mean')[name]
    df_train[name + '_trend'] = np.divide(df_train[name], df_train[name+'_day'])
    trend_feats.append(name+'_trend')

df_all = df_train.groupby(['day']).agg('mean')['temp_in']
print(df_all)
his_mean_day = df_all.mean()
record = df_train.groupby(['hour', 'minute']).agg('median').reset_index()[['hour', 'minute'] + trend_feats]
record.columns = ['hour', 'minute', 'his_temp_in']
record.loc[~record['hour'].isin([0, 12]), 'his_temp_in'] = record.loc[~record['hour'].isin([0, 12]), 'his_temp_in'].rolling(window=8, center=True, min_periods=1).mean()
record['his_temp_in'] = record['his_temp_in'] * his_mean_day

# 历史值
df_train1 = df_train.drop(index=df_train[df_train['day'].isin([19,20])].index)
record1 = df_train1.groupby(['hour', 'minute']).agg('mean').reset_index()[['hour', 'minute', 'temp_in']]
record1.columns = ['hour', 'minute', 'his_temp_in']
for name in ['his_temp_in']:
    record1.loc[~record1['hour'].isin([0, 12]), name] = record1.loc[~record1['hour'].isin([0, 12]), name].rolling(window=10, center=True, min_periods=1).mean()
record['his_temp_in'] = np.add(record['his_temp_in']*0.3, record1['his_temp_in']*0.7)

df_sample = df_sample.merge(record, on=['hour', 'minute'], how='left')
df_sample['his_temp_in'] = df_sample['his_temp_in'].fillna(method='ffill')

# 根据起始时刻填入历史值
tmp = np.subtract(df_sample['temp_in'], df_sample['his_temp_in'])
for name in ['temp_in']:
    df_sample.loc[11:, name] = df_sample.index[11:].map(lambda x: np.add(tmp.loc[x-(x-10)%120],
                                                                         df_sample.loc[x, 'his_'+name]))
# 进行趋势修改
for name in ['temp_in']:
    for i in range(1, 47):
        idx = i * 120 + 10
        k1 = (df_sample.loc[idx+119, name]-df_sample.loc[idx, name])/119.  # 填充本期温度斜率
        k2 = (df_sample.loc[idx+120, 'future_'+name]-df_sample.loc[idx, name])/120.  # 预测本期斜率

        k11 = (df_sample.loc[idx, name] - df_sample.loc[idx - 120, name]) / 120.  # 实际上期温度斜率
        k21 = (df_sample.loc[idx, 'his_' + name] - df_sample.loc[idx - 120, 'his_' + name]) / 120.  # 历史上期温度斜率

        df_sample.loc[idx:119 + idx, name] = df_sample.loc[idx:119 + idx, name] + (
                range(120) * (k2 - k1) * 1. * 0.85 + range(120) * (k11 - k21) * (-1.2) * 0.15
        )


df_sample.loc[:9, 'temp_in'] = df_sample.loc[:9, 'his_temp_in'].values

# test集时刻使用了当前时刻真实值，置为nan后ffill填充
idx = df_sample.loc[df_sample['flag']==1].index
df_sample.loc[idx, 'temp_in'] = np.nan
df_sample['temp_in'] = df_sample['temp_in'].fillna(method='ffill')-0.05

df_sample[['time', 'temp_in']].to_csv('submit_ans/predict_first.csv', header=['time', 'temperature'], index=False)
