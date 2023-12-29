import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys, os

# 親ディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.point_history_utils import replace_nan, set_dtype, parse_date

# 浮動小数点数を小数点以下3桁で表示するように設定
pd.set_option('display.float_format', '{:.3f}'.format)




if __name__ == '__main__':
    #concat_csv()
    df = pd.read_csv('data/input/point_history.csv', encoding="utf-8")
    df = replace_nan(df)
    df = set_dtype(df)
    df['super'] = df['super'].str.replace(r'\s+', '', regex=True)
    #print(df['リサイクル分類ID'].unique())
    #df = df[(df['リサイクル分類ID'] == "1") | (df['リサイクル分類ID'] == "1.0") | (df['リサイクル分類ID'] == np.nan)]  # 古紙データとポイント利用データを抽出
    #show_total_recycle_amount_per_date_noncleansing(df)
    #aggregate_shop_date_noncleansing(df)

    #extract_one_shop(df, 'ヨークベニマル', '佐野田島町店')
    #extract_one_shop(df, 'ヨークベニマル', '南中山店')
    #extract_one_shop(df, 'ヨークベニマル', '西那須野店')
    #extract_one_shop(df, 'ヨークベニマル', '若松原店')
    #extract_one_shop(df, 'ビフレ', '東通店')
    extract_one_shop(df, 'みやぎ生協', '加賀野店')
    extract_one_shop(df, 'みやぎ生協', '石巻大橋店')
