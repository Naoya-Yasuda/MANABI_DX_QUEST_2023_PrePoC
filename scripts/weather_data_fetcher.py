
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys, os

# 親ディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.point_history_utils import replace_nan, set_dtype, parse_date


df_point_history = pd.read_csv('data/input/point_history.csv', encoding="utf-8")  # point_history_2.csv（都道府県、市を含む）を読み込む
df_weather = pd.read_csv('data/input/weather_data/weather.csv', encoding="utf-8")

df_point_history = replace_nan(df_point_history)
df_weather = replace_nan(df_weather)

# prefectures列の値がnanの行を削除
df_point_history = df_point_history[pd.notna(df_point_history["prefectures"])]

# use_date列をparse_date関数で日付型に変換し、時間は切り捨てし、[use_date_2]列に格納
df_point_history['use_date_2'] = pd.to_datetime(df_point_history['use_date']).dt.floor('d')

# '市'または'群'に続く文字を削除
df_point_history['municipality'] = df_point_history['municipality'].str.replace(r'(市|郡).*', r'\1', regex=True)
df_point_history.loc[df_point_history['municipality'] == "群"].loc['municipality'] = "郡山群"
df_point_history.loc[df_point_history['municipality'] == "塩竉市"].loc['municipality'] = "塩竈市"
# 'use_date_2'列をdatetime型に変換
df_point_history['use_date_2'] = pd.to_datetime(df_point_history['use_date_2'])
# '年月日'列をdatetime型に変換
df_weather['年月日'] = pd.to_datetime(df_weather['年月日'])
# df_weatherのlat, lon列を削除
df_weather = df_weather.drop(columns=['lat', 'lon'])

# df_point_historyのmunicipalityとuse_date_2、df_weatherの市と年月日で結合し、df_point_history_weatherに格納
df_point_history_weather = pd.merge(df_point_history, df_weather, left_on=['municipality', 'use_date_2'], right_on=['市', '年月日'], how='left')

# 列削除
df_point_history_weather = df_point_history_weather.drop(columns=['use_date_2'])
df_point_history_weather = df_point_history_weather.drop(columns=['県','市'])
df_point_history_weather = df_point_history_weather[(df_point_history_weather["天気"] == "晴") | (df_point_history_weather["天気"] == "曇") | (df_point_history_weather["天気"] == "雨") | (df_point_history_weather["天気"] == "雪")]
df_point_history_weather = df_point_history_weather[pd.notna(df_point_history_weather["amount_kg"])]

print("shape",df_point_history_weather.shape)
print(df_point_history_weather["天気"].value_counts())
print("天気nan例",df_point_history_weather[(df_point_history_weather["天気"] != "晴") & (df_point_history_weather["天気"] != "曇") & (df_point_history_weather["天気"] != "雨") & (df_point_history_weather["天気"] != "雪")][:10])

df_point_history_weather.to_csv('data/input/point_history_weather.csv', index=False, encoding="utf-8")




