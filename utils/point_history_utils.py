import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def replace_nan(df):
    """
    Replace 'N', 'NaN', 'nan', 'foo', '///' to np.nan
    args:
        df: pandas.DataFrame
    return:
        df: pandas.DataFrame    
    """
    df = df.replace('N', np.nan)
    df = df.replace('NaN', np.nan)
    df = df.replace('nan', np.nan)
    df = df.replace('foo', np.nan)
    df = df.replace('///', np.nan)
    return df

def parse_date(date):
    """
    Parse date string to datetime object
    args:
        date: str
    return:
        datetime object
    """
    try:
        return pd.to_datetime(date)
    except ValueError:
        try:
            return pd.to_datetime(date, format='%Y年%m月%d日')
        except ValueError:
            return pd.to_datetime(date, format='%Y/%m/%d')




def set_dtype(df):
    # 列名を直感的に変更
    df = df.rename(columns={'id_1': '支店ID'})
    df = df.rename(columns={'item_id': 'リサイクル分類ID'})

    column_types = {
        'id':np.float32,
        'user_id':np.float64,
        'series_id' : np.float32,
        'shop_id' : str,
        'shop_name' : str,
        'card_id' : str,
        'リサイクル分類ID' : str,
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