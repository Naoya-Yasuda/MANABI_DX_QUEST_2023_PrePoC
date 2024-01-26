import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from dateutil.relativedelta import relativedelta

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
    column_types = {col: dtype for col, dtype in column_types.items() if col in df.columns} # dfに存在するカラムのみ型変換を適用
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


def open_point_history_per_shop(super, shop_name_1): 
    """
    店舗ごとに分けたpoint_historyファイルを開く関数
    args:
        super: スーパー名
        shop_name_1: 店舗名
    return:
        df: pandas.DataFrame
    """

    df = pd.read_csv(f'data/input/shop_data/point_history_{super}_{shop_name_1}.csv', encoding="utf-8")
    df = set_dtype(df)
    df = replace_nan(df)
    return df

def aggregate_shop_date(df):
    """
    point_historyをshop_id、日付ごとにグループ化し、合計値と代表値を計算する関数
    args:
        df: データフレーム
    return:
        aggregated_df: データフレーム
    """

    df = set_dtype(df)
    df = replace_nan(df)

    aggregation_columns = {
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
        '合計全天日射量(MJ/㎡)': 'first',
        'interval_compared_to_previous': 'first',
        'interval_compared_to_next': 'first',
    }

    # 存在する列のみを選択
    existing_columns = {col: agg_method for col, agg_method in aggregation_columns.items() if col in df.columns}

    # shop_idと年月日ごとにグループ化し、合計値と代表値を計算
    aggregated_df = df.groupby(['shop_id', '年月日']).agg(existing_columns).reset_index()

    # shop_idと年月日でソート
    aggregated_df = aggregated_df.sort_values(by=['shop_id', '年月日'])

    return aggregated_df

def aggregate_date(df):
    """
    point_historyを日付ごとにグループ化し、合計値と代表値を計算する関数
    args:
        df: データフレーム
    return:
        aggregated_df: データフレーム
    """

    df = set_dtype(df)
    df = replace_nan(df)

    # 集計する列を定義
    aggregation = {
        'amount': 'sum',
        'amount_kg': 'sum',
        'point': 'sum',
        'total_point': 'sum',
        'total_amount': 'sum',
        'coin': 'sum',
        'shop_id' : 'first',
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
        '合計全天日射量(MJ/㎡)': 'first',
        'interval_compared_to_previous': 'max',
        'interval_compared_to_next': 'max',
    }

    # 'filling_rate' 列が存在する場合のみ追加
    if 'filling_rate' in df.columns:
        aggregation['filling_rate'] = 'max'

    # 'total_amount_kg_per_day' 列が存在する場合のみ追加
    if 'total_amount_kg_per_day' in df.columns:
        aggregation['total_amount_kg_per_day'] = 'max'

    # shop_idと年月日ごとにグループ化し、合計値と代表値を計算
    aggregated_df = df.groupby(['年月日']).agg(aggregation).reset_index()
    aggregated_df = aggregated_df.sort_values(by=['年月日'])
    
    return aggregated_df




# 日付特徴量の追加
def add_date_features(df):
    df = df.copy()
    df["month"] = df["年月日"].dt.month
    df["day"] = df["年月日"].dt.day
    df["year"] = df["年月日"].dt.year
    df['day_of_week'] = df['年月日'].dt.day_name()

    df["day_sin"] = np.sin(df["day"] / 31 * 2* np.pi)
    df["day_cos"] = np.cos(df["day"] / 31 * 2* np.pi)
    df.drop(columns=["day"], inplace=True)
    
    df["month_sin"] = np.sin(df["month"] / 12 * 2* np.pi)
    df["month_cos"] = np.cos(df["month"] / 12 * 2* np.pi)
    df.drop(columns=["month"], inplace=True)
    return df

def set_previous_data(df, features, days=28, years=0):
    """
    指定された日数または年数前の特徴量の値を取得する関数。
    ※年数と日数のどちらか一方のみ指定可能。
    args:
        df: データフレーム
        features: 特徴量のリスト
        days: 日数（デフォルトは28）
        years: 年数（デフォルトは0）
    return:
        df: 更新されたデータフレーム
    """
    # 日付の計算
    if years > 0:
        df['date_previous'] = df['年月日'].apply(lambda x: x - relativedelta(years=years))
        time_label = str(years) + 'years'
    else:
        df['date_previous'] = df['年月日'] - pd.Timedelta(days=days)
        time_label = str(days) + 'days'

    for feature in features:
        new_feature = feature + '_before_' + time_label
        # 一時的なデータフレームを作成
        temp_df = df[['年月日', 'super', 'shop_name_1', feature]].copy()
        temp_df.rename(columns={'年月日': 'date_previous', feature: new_feature}, inplace=True)

        # 元のデータフレームに一時的なデータフレームをマージ
        df = df.merge(temp_df, on=['super', 'shop_name_1', 'date_previous'], how='left')

    # 不要な列を削除
    df.drop('date_previous', axis=1, inplace=True)

    return df