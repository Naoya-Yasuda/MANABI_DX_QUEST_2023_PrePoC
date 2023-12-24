import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def replace_nan(df):
    df = df.replace('N', np.nan)
    df = df.replace('NaN', np.nan)
    df = df.replace('nan', np.nan)
    df = df.replace('foo', np.nan)
    df = df.replace('///', np.nan)
    return df

def set_dtype(df):
    column_types = {
        'id':np.float32,
        'user_id':np.float64,
        'series_id' : np.float32,
        'shop_id' : str,
        'shop_name' : str,
        'card_id' : str,
        'amount' : np.float32,
        'amount_kg' : np.float32,
        'point' : np.float32,
        'total_point' : np.float32,
        'total_amount' : np.float32,
        'coin' : np.float32,
        'rank_id':np.float32,
        'use_date': 'datetime64[ns]',
        'created_at': 'datetime64[ns]',
        'updated_at': 'datetime64[ns]',
        '支店ID' : np.float32,
        'super' : str,
        'prefectures' : str,
        'municipality' : str,
        'shop_name_1' :  str,
        'shop_id_1' :    str,
        'store_latitude' : np.double,
        'store_longitude' : np.double,
        '年月日' : 'datetime64[ns]',
        '天気': str,
        '平均気温(℃)': np.float32,
        '最高気温(℃)': np.float32,
        '最低気温(℃)': np.float32,
        '降水量の合計(mm)': np.float32,
        '平均風速(m/s)': np.float32,
        '平均湿度(％)': np.float32,
        '平均現地気圧(hPa)': np.float32,
        '平均雲量(10分比)': np.float32,
        '降雪量合計(cm)': np.float32,
        '日照時間(時間)': np.float32,
        '合計全天日射量(MJ/㎡)': np.float32,
    }
    df = df.astype(column_types)
    return df

def aggregate_shop_date(df):
    # shop_idと年月日ごとにグループ化し、合計値と代表値を計算
    aggregated_df = df.groupby(['shop_id', '年月日']).agg({
        'amount': 'sum',
        'amount_kg': 'sum',
        'point': 'sum',
        'total_point': 'sum',
        'total_amount': 'sum',
        'coin': 'sum',
        'series_id': 'first',
        'shop_name': 'first',
        'リサイクル分類ID': 'first',
        '支店ID': 'first',
        'super': 'first',
        'prefectures': 'first',
        'municipality': 'first',
        'shop_name_1': 'first',
        'shop_id_1': 'first',
        'store_opening_time': 'first',
        'store_closing_time': 'first',
        'rps_opening_time': 'first',
        'rps_closing_time': 'first',
        'store_latitude': 'first',
        'store_longitude': 'first',
        '天気': 'first',
        '平均気温(℃)': 'first',
        '最高気温(℃)': 'first',
        '最低気温(℃)': 'first',
        '降水量の合計(mm)': 'first',
        '平均風速(m/s)': 'first',
        '平均湿度(％)': 'first',
        '平均現地気圧(hPa)': 'first',
        '平均雲量(10分比)': 'first',
        '降雪量合計(cm)': 'first',
        '日照時間(時間)': 'first',
        '合計全天日射量(MJ/㎡)': 'first'
    }).reset_index()

    # shop_idと年月日でソート
    aggregated_df = aggregated_df.sort_values(by=['shop_id', '年月日'])

    # 結果を保存
    aggregated_df.to_csv('data/input/point_history_per_shop_date.csv', index=False, encoding="utf-8")

if __name__ == '__main__':
    df = pd.read_csv('data/input/point_history_rps_weather.csv', encoding="utf-8")
    df = replace_nan(df)
    df = set_dtype(df)
    aggregate_shop_date(df)