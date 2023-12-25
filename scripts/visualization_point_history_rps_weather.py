import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 浮動小数点数を小数点以下3桁で表示するように設定
pd.set_option('display.float_format', '{:.3f}'.format)

# Windows MatplotlibのデフォルトフォントをMeiryoに設定
#plt.rcParams['font.family'] = 'Meiryo'

# Mac Matplotlibのデフォルトフォントをヒラギノ角ゴシックに設定
plt.rcParams['font.family'] = 'Hiragino Sans'

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

def aggregate_date(df):
    # shop_idごとにグループ化し、合計値と代表値を計算
    aggregated_df = df.groupby(['shop_id']).agg({
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

    # shop_idでソート
    aggregated_df = aggregated_df.sort_values(by=['shop_id'])

    # 結果を保存
    aggregated_df.to_csv('data/input/point_history_per_shop.csv', index=False, encoding="utf-8")

# カスタム関数を定義
def parse_date(date):
    try:
        return pd.to_datetime(date)
    except ValueError:
        try:
            return pd.to_datetime(date, format='%Y年%m月%d日')
        except ValueError:
            return pd.to_datetime(date, format='%Y/%m/%d')

def replace_nan(df):
    df = df.replace('N', np.nan)
    df = df.replace('NaN', np.nan)
    df = df.replace('nan', np.nan)
    df = df.replace('foo', np.nan)
    df = df.replace('///', np.nan)
    return df

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
        #'年月日' : 'datetime64[ns]',
        #'天気': str,
        #'平均気温(℃)': np.float32,
        #'最高気温(℃)': np.float32,
        #'最低気温(℃)': np.float32,
        #'降水量の合計(mm)': np.float32,
        #'平均風速(m/s)': np.float32,
        #'平均湿度(％)': np.float32,
        #'平均現地気圧(hPa)': np.float32,
        #'平均雲量(10分比)': np.float32,
        #'降雪量合計(cm)': np.float32,
        #'日照時間(時間)': np.float32,
        #'合計全天日射量(MJ/㎡)': np.float32,
    }
    df = df.astype(column_types)
    return df

def show_total_recycle_amount_per_date_noncleansing(df):
    # Nanに置き換え
    df = replace_nan(df)

    # 型変換
    df = set_dtype(df)

    # use_date列をparse_date関数で日付型に変換し、時間は切り捨てし、[use_date_2]列に格納
    df['年月日'] = pd.to_datetime(df['use_date']).dt.floor('d')

    # '年月日'でグループ化し、'amount_kg'の合計値を計算
    df_sum = df.groupby('年月日')['amount_kg'].sum().reset_index()
    fig, ax = plt.subplots()
    ax.plot(df_sum["年月日"], df_sum["amount_kg"], label='all data', color='blue', alpha=0.5)

    # ぐるっとポン未利用者を除外
    # user_id列がNanでない行のみ抽出
    df2 = df[df['user_id'].notnull()]
    print(df.shape)
    print(df2.shape)
    df2_sum = df2.groupby('年月日')['amount_kg'].sum().reset_index()
    ax.plot(df2_sum["年月日"], df2_sum["amount_kg"], label='ぐるっとポンユーザ', color='red', alpha=0.5)

    # x軸のラベル表示間隔を調整
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.set_xlabel('年月日')
    ax.set_ylabel('total recycle amount[kg]')

    ax.set_yscale('log')
    ax.legend()
    plt.savefig('data/input/total_recycle_amount_per_date_noncleansing.png')
    plt.show()

def aggregate_shop_date_noncleansing(df):
    # Nanに置き換え
    df = replace_nan(df)

    # 型変換
    df = set_dtype(df)
    
    # use_date列をparse_date関数で日付型に変換し、時間は切り捨てし、[use_date_2]列に格納
    df['年月日'] = pd.to_datetime(df['use_date']).dt.floor('d')

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
    }).reset_index()

    # shop_idと年月日でソート
    aggregated_df = aggregated_df.sort_values(by=['shop_id', '年月日'])

    # 結果を保存
    aggregated_df.to_csv('data/input/point_history_per_shop_date_noncleansing.csv', index=False, encoding="utf-8")

def concat_csv():
    # data/input/point_history_1.csv ~ data/input/point_history_15.csv を結合。data/input/point_history.csvに保存
    df = pd.concat([pd.read_csv(f'data/input/point_history_{i}.csv', encoding="utf-8") for i in range(1, 16)])
    # 列削除
    df = df.drop(columns=['shop_url', 'free_text', 'rps_target_store', 'collect_item', 'record_id',
                          'other_shop_id', 'deactivated_flg', 'is_search_result_display'])
    df = df.replace('N', np.nan)
    df['use_date'] = pd.to_datetime(df['use_date'], errors='coerce')
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
    df['created_at_1'] = pd.to_datetime(df['created_at_1'], errors='coerce')
    df['updated_at_1'] = pd.to_datetime(df['updated_at_1'], errors='coerce')
    # time型に変換
    df['store_opening_time'] = pd.to_datetime(
        df['store_opening_time'], format='%H:%M:%S').dt.time
    df['store_closing_time'] = pd.to_datetime(
        df['store_closing_time'], format='%H:%M:%S').dt.time
    df['rps_opening_time'] = pd.to_datetime(
        df['rps_opening_time'], format='%H:%M:%S').dt.time
    df['rps_closing_time'] = pd.to_datetime(
        df['rps_closing_time'], format='%H:%M:%S').dt.time

    df = set_dtype(df)
    # 列名を直感的に変更
    df = df.rename(columns={'id_1': '支店ID'})
    df = df.rename(columns={'item_id': 'リサイクル分類ID'})

    df.to_csv('data/input/point_history.csv', index=False, encoding="utf-8")

    


if __name__ == '__main__':
    #concat_csv()
    df = pd.read_csv('data/input/point_history.csv', encoding="utf-8")
    df = replace_nan(df)
    df = set_dtype(df)
    print(df['リサイクル分類ID'].unique())
    df = df[(df['リサイクル分類ID'] == "1") | (df['リサイクル分類ID'] == "1.0") | (df['リサイクル分類ID'] == np.nan)]  # 古紙データとポイント利用データを抽出
    show_total_recycle_amount_per_date_noncleansing(df)
    aggregate_shop_date_noncleansing(df)