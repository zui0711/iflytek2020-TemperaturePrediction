import pandas as pd

ans1 = pd.read_csv('submit_ans/lgb_fill_first.csv')
ans2 = pd.read_csv('submit_ans/lr_fill_first.csv')
ans3 = pd.read_csv('submit_ans/predict_first.csv')

ans1['temperature'] = (ans1['temperature']*0.7+ans2['temperature']*0.3)*0.7+ans3['temperature']*0.3-0.02
ans1[['time', 'temperature']].to_csv('submit_ans/ans.csv', header=['time', 'temperature'], index=False)