import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
import time
import warnings
import lightgbm as lgb
warnings.filterwarnings("ignore")

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

# train预处理

df_train.fillna(method='ffill', inplace=True)
for name in ['press_in', 'press_out']:
    df_train.loc[(df_train[name] < 960) | (df_train[name] > 1000), name] = np.nan
for name in ['press_in', 'press_out']:
    idx = df_train.loc[df_train[name].isnull().T, name].index
    for i in idx:
        df_train.loc[i, name] = df_train.loc[i - 1:i + 1, name].mean()
# df_train.dropna(inplace=True)

df_train_without = df_train[~df_train['hour'].isin([0, 12])]
df_test_without = df_test[~df_test['hour'].isin([0, 12])]
fill_temp_out = []
ii=0
for (fill_train, fill_test) in zip([df_train, df_train_without], [df_test, df_test_without]):
    fill_feats = ['temp_out1', 'temp_out2']
    for df in [fill_train, fill_test]:
        df['humid_io_div'] = np.divide(df['humid_out'], df['humid_in'])
        df['neg_humid_io_div'] = np.divide(100 - df['humid_out'], 100 - df['humid_in'])

        df['press_io_div'] = np.divide(df['press_out'], df['press_in'])
        df['hour_time'] = df['hour'].apply(lambda x: 0 if (x < 8) | (x > 16) else 1)
        df['humid_press'] = np.add(df['humid_io_div'], df['press_io_div'])

    for name in ['temp_out', 'humid_io_div', 'press_io_div', 'hour_time', 'humid_press']:
        fill_train[name + '1'] = fill_train[name].shift(120)
        fill_train[name + '2'] = fill_train[name].shift(240)
        fill_test[name + '1'] = fill_test[name].shift(1)
        fill_test[name + '2'] = fill_test[name].shift(2)
        fill_feats.append(name + '1')
        fill_feats.append(name + '2')

    for name in ['humid_in', 'humid_out']:
        fill_train[name + '1'] = fill_train[name].shift(120)
        fill_train[name + '2'] = fill_train[name].shift(240)
        fill_test[name + '1'] = fill_test[name].shift(1)
        fill_test[name + '2'] = fill_test[name].shift(2)

    for df in [fill_train, fill_test]:
        df['temp_div'] = np.divide(df['temp_out1'], df['temp_out2'])
        df['temp_sub'] = np.subtract(df['temp_out1'], df['temp_out2'])

    fill_feats = fill_feats + ['hour', 'minute', 'temp_div', 'temp_sub']

    df1 = fill_train[fill_feats + ['temp_out', 'humid_out', 'humid_in']]
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
                'max_depth': 5-ii,
                # 'num_leaves': 2**(5-ii-1),
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
print(aaa)
ind = aaa[~aaa['temp_out_y'].isnull()].index
aaa.loc[ind, 'temp_out_x'] = aaa.loc[ind, 'temp_out_y']
df_test['future_temp_out'] = aaa['temp_out_x']

# 获取提交时间
time_list = df_sample.time.map(lambda x: [time.localtime(x).tm_mon, time.localtime(x).tm_mday, time.localtime(x).tm_hour, time.localtime(x).tm_min])
df_sample['month'] = time_list.map(lambda x: x[0])
df_sample['day'] = time_list.map(lambda x: x[1])
df_sample['hour'] = time_list.map(lambda x: x[2])
df_sample['minute'] = time_list.map(lambda x: x[3])

# 加入测试集数据
df_sample = df_sample.merge(df_test[original_feats+['time', 'flag', 'future_temp_out']], on=['time'], how='left')

# 加入训练集历史均值
# 周期因子
trend_feats = []
for name in original_feats:
    df_train[name+'_day'] = df_train.groupby(['day']).transform('mean')[name]
    df_train[name + '_trend'] = np.divide(df_train[name], df_train[name+'_day'])
    trend_feats.append(name+'_trend')

df_all = df_train.groupby(['day']).agg('mean')[original_feats]
his_mean_day = df_all.mean()
print(his_mean_day)

record = df_train.groupby(['hour', 'minute']).agg('median').reset_index()[['hour', 'minute'] + trend_feats]
record.columns = ['hour', 'minute'] + his_feats
for name, name1 in zip(his_feats, original_feats):
    record.loc[~record['hour'].isin([0, 12]), name] = record.loc[~record['hour'].isin([0, 12]), name].rolling(window=8, center=True, min_periods=1).mean()
    record[name] = record[name] * his_mean_day[name1]

# 历史值
df_train1 = df_train.drop(index=df_train[df_train['day'].isin([19,20])].index)
record1 = df_train1.groupby(['hour', 'minute']).agg('mean').reset_index()[['hour', 'minute'] + original_feats]
record1.columns = ['hour', 'minute'] + his_feats
for name in his_feats:
    record1.loc[~record1['hour'].isin([0, 12]), name] = record1.loc[~record1['hour'].isin([0, 12]), name].rolling(window=10, center=True, min_periods=1).mean()

record[his_feats] = np.add(record[his_feats]*0.35, record1[his_feats]*0.65)
df_sample = df_sample.merge(record, on=['hour', 'minute'], how='left')

# 空缺为12点0/1分，对历史数据bfill不穿越
df_sample[his_feats] = df_sample[his_feats].fillna(method='bfill')

# 利用每2小时数据及历史均值计算
tmp = np.subtract(df_sample[original_feats], df_sample[his_feats])
for name in original_feats:
    df_sample.loc[11:, name] = df_sample.index[11:].map(lambda x: np.add(tmp.loc[x-(x-10)%120, name],
                                                                         df_sample.loc[x, 'his_'+name]))

for name in ['temp_out']:
    for i in range(1, 47):
        idx = i * 120 + 10

        k1 = (df_sample.loc[idx+119, name]-df_sample.loc[idx, name])/119.  # 填充本期温度斜率
        k2 = (df_sample.loc[idx+120, 'future_'+name]-df_sample.loc[idx, name])/120.  # 预测本期斜率

        k11 = (df_sample.loc[idx, name]-df_sample.loc[idx-120, name])/120.  # 实际上期温度斜率
        k21 = (df_sample.loc[idx, 'his_'+name]-df_sample.loc[idx-120, 'his_'+name])/120.  # 历史上期温度斜率

        df_sample.loc[idx:119+idx, name] = df_sample.loc[idx:119+idx, name] + (
                range(120)*(k2-k1)*1. * 0.8 + range(120)*(k11-k21)*(-1.1) * 0.2
        )

# 利用历史均值补充前10个数据
df_sample.loc[:9, original_feats] = df_sample.loc[:9, his_feats].values

feats = original_feats+['hour', 'minute']

for df in [df_train, df_sample]:
    df['hour1'] = df['hour'].apply(lambda x: 0 if (x < 8) | (x > 16) else 1)
    df['humid_div'] = np.divide(df['humid_in'], df['humid_out'])
    df['humid_div1'] = np.divide(100 - df['humid_in'], 100 - df['humid_out'])
    df['humid_temp_out'] = df['humid_div'] * df['temp_out']

feats = feats + ['hour1', 'humid_div', 'humid_div1', 'humid_temp_out']

l1 = len(df_train)
df_all = pd.concat([df_train, df_sample], axis=0, ignore_index=True)

for name in original_feats:
    df_all[name + '_halfhour'] = df_all[name].shift(30)
    df_all[name + '_onehour'] = df_all[name].shift(60)
    df_all[name + '_twohour'] = df_all[name].shift(120)

    df_all[name + '_halfhour_diff'] = df_all[name] - df_all[name + '_halfhour']
    df_all[name + '_onehour_diff'] = df_all[name + '_halfhour'] - df_all[name + '_onehour']
    df_all[name + '_twohour_diff'] = df_all[name + '_onehour'] - df_all[name + '_twohour']

    df_all[name + '_halfhour_div'] = np.divide(df_all[name], df_all[name + '_halfhour'])
    df_all[name + '_onehour_div'] = np.divide(df_all[name + '_halfhour'], df_all[name + '_onehour'])
    df_all[name + '_twohour_div'] = np.divide(df_all[name + '_onehour'], df_all[name + '_twohour'])
for name in ['temp_out', 'humid_out', 'humid_in']:
    feats = feats + [name + '_halfhour', name + '_onehour', name + '_twohour']
    feats = feats + [name + '_halfhour_diff', name + '_onehour_diff']
    feats = feats + [name + '_halfhour_div', name + '_onehour_div']
for name in ['press_out', 'press_in']:
    feats = feats + [name + '_halfhour', name + '_onehour', name + '_twohour']

for name in ['humid_temp_out', 'humid_div', 'humid_div1']:
    df_all[name + '_halfhour'] = df_all[name].shift(30)
    df_all[name + '_onehour'] = df_all[name].shift(60)
    df_all[name + '_halfhour_diff'] = df_all[name] - df_all[name + '_halfhour']
    df_all[name + '_onehour_diff'] = df_all[name + '_halfhour'] - df_all[name + '_onehour']
    df_all[name + '_halfhour_div'] = np.divide(df_all[name], df_all[name + '_halfhour'])
    df_all[name + '_onehour_div'] = np.divide(df_all[name + '_halfhour'], df_all[name + '_onehour'])

for name in ['humid_temp_out', 'humid_div', 'humid_div1']:
    feats = feats + [name + '_halfhour', name + '_onehour']

df_train = df_all[:l1]
df_sample = df_all[l1:]

df_sample[feats] = df_sample[feats].fillna(method='ffill')

train_tmp = df_train[feats+['temp_in']].dropna()
print(train_tmp.shape)
trainx = train_tmp[feats]
trainy = train_tmp['temp_in']

scalerx = StandardScaler()
trainx = scalerx.fit_transform(trainx)
testx = scalerx.transform(df_sample[feats])

model = Ridge(alpha=0.001)
model.fit(trainx, trainy)
pred_trainy = model.predict(trainx)
print(mse(trainy, pred_trainy))

y = model.predict(testx)
c = model.coef_

df_sample['temperature'] = y
idx = df_sample.loc[df_sample['flag']==1].index
df_sample.loc[idx, 'temperature'] = np.nan
df_sample['temperature'] = df_sample['temperature'].fillna(method='ffill')

print(df_sample['temperature'])
df_sample['temperature'] = df_sample['temperature']-0.08
df_sample[['time', 'temperature']].to_csv('submit_ans/lr_fill_first.csv', header=['time', 'temperature'], index=False)