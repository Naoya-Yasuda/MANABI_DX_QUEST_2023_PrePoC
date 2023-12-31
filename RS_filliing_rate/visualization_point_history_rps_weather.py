import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys, os

# 親ディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.point_history_utils import replace_nan, set_dtype, parse_date

# 浮動小数点数を小数点以下2桁で表示するように設定
pd.set_option('display.float_format', '{:.2f}'.format)

# Windows MatplotlibのデフォルトフォントをMeiryoに設定
#plt.rcParams['font.family'] = 'Meiryo'

# Mac Matplotlibのデフォルトフォントをヒラギノ角ゴシックに設定
#plt.rcParams['font.family'] = 'Hiragino Sans'

# クラウド環境用
plt.rcParams['font.family'] = 'Noto Sans CJK JP'


def aggregate_date(df):
    """
    point_historyをshop_idごとにグループ化し、合計値と代表値を計算し保存
    args:
        df: データフレーム
    return:
        None
    """
    # 
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

def show_total_recycle_amount_per_date(df):
    """
    日付ごとの総リサイクル量をグラフに表示
    args:
        df: データフレーム
    return:
        None
    """
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
    plt.savefig('data/input/total_recycle_amount_per_date.png')
    plt.show()




if __name__ == '__main__':
    df = pd.read_csv('data/input/point_history_weather.csv', encoding="utf-8")
    df = replace_nan(df)
    df = set_dtype(df)
    df['super'] = df['super'].str.replace(r'\s+', '', regex=True)
    df = df[(df['リサイクル分類ID'] == "1") | (df['リサイクル分類ID'] == "1.0") | (df['リサイクル分類ID'] == np.nan)]  # 古紙データとポイント利用データを抽出
    show_total_recycle_amount_per_date(df)
    # aggregate_shop_date(df)

    #extract_one_shop(df, 'ヨークベニマル', '佐野田島町店')
    #extract_one_shop(df, 'ヨークベニマル', '南中山店')
    #extract_one_shop(df, 'ヨークベニマル', '西那須野店')
    #extract_one_shop(df, 'ヨークベニマル', '若松原店')
    #extract_one_shop(df, 'ビフレ', '東通店')
    #extract_one_shop(df, 'みやぎ生協', '加賀野店')
    #extract_one_shop(df, 'みやぎ生協', '石巻大橋店')

    