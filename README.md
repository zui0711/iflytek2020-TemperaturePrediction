# iflytek2020-TemperaturePrediction

2020年讯飞开发者大赛温室温度赛道 复赛No.3解决方案

比赛链接 http://challenge.xfyun.cn/topic/info?type=temperature

成绩A榜0.07826，B榜成绩0.07724

代码包含4部分
1. lgb_fll_first.py 先填充中间特征，再利用lgb建模预测室内温度，A榜成绩0.07995
2. lr_fll_first.py 先填充中间特征，再利用Ridge(实际正则取很小，约等于LinearRegression)建模预测室内温度，A榜成绩0.08412
3. predict_first.py 先预测每两小时室内温度(LR+lgb融合)，再对中间时刻室内温度进行填充，A榜成绩0.08468
4. aver.py 对以上3个答案进行加权平均及施加偏置获得最后结果

更多赛题解读细节见
