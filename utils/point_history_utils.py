import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


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
    """
    set dtype of each column
    args:
        df: pandas.DataFrame
    return:
        df: pandas.DataFrame
    """

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
        'rps_opening_time' : 'datetime64[ns]',
        'rps_closing_time' : 'datetime64[ns]',
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
        'interval_compared_to_previous': np.float32,
        'interval_compared_to_next': np.float32,
    }
    df = df.astype(column_types)
    return df



def replace_nan(df):
    """
    注意：set_dtypeの後に実行してください
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


def delete_blank_from_filename(directory):
    """
    ファイル名から空白を除去する関数
    args:
        directory: ディレクトリ名
    return:
        None
    """
    # 対象のディレクトリを設定
    directory = 'data/input/shop_data/'

    # ディレクトリ内の全ファイルをチェック
    for filename in os.listdir(directory):
        # 新しいファイル名を生成（空白を除去）
        new_filename = filename.replace(' ', '')
        # 元のファイル名と新しいファイル名のフルパスを取得
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # ファイル名を変更
        os.rename(old_file, new_file)
        print(f"Renamed '{old_file}' to '{new_file}'")