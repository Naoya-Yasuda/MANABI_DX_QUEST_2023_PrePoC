from itertools import islice
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from tqdm import tqdm 
import sys
from datetime import datetime, timedelta, time
from scipy.optimize import curve_fit
from scipy import stats

# 親ディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.point_history_utils import replace_nan, set_dtype, parse_date

# 浮動小数点数を小数点以下2桁で表示するように設定
pd.set_option('display.float_format', '{:.2f}'.format)

# べき乗則関数を定義
def power_law(x, a, b):
    return a * np.power(x, b)

# 指数関数を定義
def exp_func(x, a, b):
    return a*np.exp(-b*x)

def calc_recycle_period(df, super_name, shop_name):
    """
    リサイクルステーションの利用間隔を計算する関数
    args:
        df: データフレーム
        super_name: スーパー名
        shop_name: 店舗名
    return:
        interval: 利用間隔のリスト
    """
    df2 = df[(df['super'] == super_name) & (df['shop_name_1'] == shop_name)].sort_values('use_date')
    # use_date列の差分を計算
    df2['interval'] = df2['use_date'].diff()

    # df['年月日']について前の行と日付が異なる場合、df['rps_closing_time']とdf['rps_opening_time']の差をdf['interval']に格納
    df2.loc[df['年月日'].diff().dt.total_seconds() != 0, 'interval'] -= df2['rps_closing_time'] - df2['rps_opening_time']


    df2['interval'] = df2['interval'].dt.total_seconds() / 3600

    # 最初の行には nan を設定
    df2.loc[0, 'interval'] = np.nan

    return df2['interval']

def plot_recycle_period(interval, super_name, shop_name, ax, func):
    """
    リサイクルステーションの利用間隔のヒストグラムをプロットする関数
    args:
        interval: 利用間隔のリスト
        super_name: スーパー名
        shop_name: 店舗名
        ax: サブプロット
        func: フィット関数
    return:
        counts: ヒストグラムの度数
        params: フィット関数のパラメータ
        bin_edges: ビンの境界
        bin_centers: ビンの中心
    """
    # ヒストグラムのデータを取得
    counts, bin_edges = np.histogram(interval, bins=100, range=(0, 12))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # x軸の大きい値を重視してべき乗則のフィットを行う
    mask = counts > 0
    weights = 1 / bin_centers[mask]

    # 重み付きフィットを実行
    try:
        initial_guess = [11.95740079,  1.03682929]
        params, params_covariance = curve_fit(func, bin_centers[mask], counts[mask], sigma=weights,  maxfev=1000, p0=initial_guess)
    except RuntimeError as err:
        print("Optimal parameters not found. Using the last tried parameters.")
        params = [11.95740079,  1.03682929]    # 仮の値を設定

    # フィット結果をプロット
    ax.bar(bin_centers, counts, width=np.diff(bin_edges), label='Data')
    ax.plot(bin_centers, func(bin_centers, *params)+0.1, label='Fit: a=%.2f, b=%.2f' % tuple(params), color='red')
    ax.set_xlabel( "Interval of Use for \n Recycling Station [h]" )
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    ax.set_yscale('log')

    return counts, params, bin_edges, bin_centers

# べき乗則関数を定義
def power_law(x, a, b):
    """
    return : a * np.power(x, b)
    """
    return a * np.power(x, b)

# 指数関数を定義
def exp_func(x, a, b):
    """
    return : a * np.exp(-b*x)
    """
    return a*np.exp(-b*x)

def chi_squared_statistic(func, params, bin_centers, counts):
    """
    カイ二乗統計量を計算する関数
    arges:
        func: フィット関数
        params: フィット関数のパラメータ
        bin_centers: 各ビンの中心値
        counts: 各ビンの度数
    return:
        p_value: p値
    """
    a, b = params[:2]
    expected = func(bin_centers, a, b)
    #expected = a * np.power(bin_centers, b) * (bin_centers[1] - bin_centers[0])  # 各ビンでの期待頻度

    index = 5   # 利用間隔が短いものは、重要でない＋値が大きく影響が大きいため、解析対象から省く
    expected *= sum(counts[index:]) / sum(expected[index:])  # 期待頻度を正規化
    # カイ二乗適合度検定
    chi_squared_stat, p_value = stats.chisquare(counts[index:], f_exp=expected[index:])

    return p_value

if __name__ == '__main__':
    df = pd.read_csv('data/input/point_history_weather.csv', encoding="utf-8")    

    # 前処理
    df = set_dtype(df)
    df = replace_nan(df)
    df['super'] = df['super'].fillna('イトーヨーカドー')
    df['rps_opening_time'] = pd.to_datetime(df['use_date'].dt.date.astype(str) + ' ' + df['rps_opening_time'])
    df['rps_closing_time'] = pd.to_datetime(df['use_date'].dt.date.astype(str) + ' ' + df['rps_closing_time'])

    # dfのsuper列とshop_name_1列の組み合わせ一覧
    df_shop_list = df[['super', 'shop_name_1']].drop_duplicates().reset_index(drop=True)


    df_shop_list['p_value'] = np.nan
    num_shop = len(df_shop_list)
    for i in tqdm(range(num_shop)):
        super_name = df_shop_list['super'][i]
        shop_name = df_shop_list['shop_name_1'][i]
        df_tmp = df[(df['super'] == super_name) & (df['shop_name_1'] == shop_name)].sort_values('use_date')

        # use_date列の差分を計算
        df['interval'] = df['use_date'].diff()
        # df['年月日']について前の行と日付が異なる場合、df['rps_closing_time']とdf['rps_opening_time']の差をdf['interval']に格納
        df.loc[df['年月日'].diff().dt.total_seconds() != 0, 'interval'] -= df['rps_closing_time'] - df['rps_opening_time']
        df['interval'] = df['interval'].dt.total_seconds() / 3600
        df.loc[0, 'interval'] = np.nan

        fig, ax = plt.subplots(figsize=(4, 4))
        interval = calc_recycle_period(df, super_name, shop_name)
        counts, params, bin_edges, bin_centers = plot_recycle_period(interval, super_name, shop_name, ax, exp_func)
        p_value = chi_squared_statistic(exp_func, params, bin_centers, counts)
        
        # タイトルに店舗名とIDを含める
        ax.set_title(f'p-value: {p_value:.2f}') 
        #ax.set_title(f'{super_name} {shop_name}\np-value: {p_value:.2f}') 

        # グラフを保存
        fig.tight_layout()
        fig.savefig(f'data/input/shop_data/{super_name}_{shop_name}.png', dpi=300)
        
        df_shop_list.loc[i, 'p_value'] = p_value
        df_tmp.to_csv('data/input/shop_data/point_history_' + super_name + '_' + shop_name + '.csv', encoding="utf-8")

    df_shop_list.to_csv('data/input/shop_list.csv', encoding="utf-8")