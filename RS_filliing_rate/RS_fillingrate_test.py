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
import datetime

# 親ディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.point_history_utils import replace_nan, set_dtype, parse_date

# 浮動小数点数を小数点以下2桁で表示するように設定
pd.set_option('display.float_format', '{:.2f}'.format)

def calc_recycle_period(df, super_name, shop_name):
    """
    リサイクルステーションの利用間隔を計算する関数
    args:
        df: データフレーム
        super_name: スーパー名
        shop_name: 店舗名
        shift_interval_to_previous: 利用間隔を前の行にシフトする場合はTrue

    return:
        df2: リサイクルステーションの利用間隔を追加したデータフレーム
        df2['interval_compared_to_previous']: 前回利用からの経過時間
        df2['interval_compared_to_next']: 次回利用までの経過時間
    """
    df2 = df.copy()
    df2['interval_compared_to_previous'] = df2['use_date'].diff()
    df2.loc[df2['年月日'].diff().dt.total_seconds() != 0, 'interval_compared_to_previous'] -= (df2['rps_opening_time'] - df2['rps_closing_time'].shift(1))
    df2['interval_compared_to_previous'] = df2['interval_compared_to_previous'].dt.total_seconds() / 3600
    df2.loc[0, 'interval_compared_to_previous'] = np.nan
    
    df2['interval_compared_to_next'] = df2['use_date'].diff().shift(-1)
    df2.loc[df2['年月日'].diff().shift(-1).dt.total_seconds() != 0, 'interval_compared_to_next'] -= (df2['rps_opening_time'].shift(-1) - df2['rps_closing_time'])
    df2['interval_compared_to_next'] = df2['interval_compared_to_next'].dt.total_seconds() / 3600
    df2.loc[len(df2)-1, 'interval_compared_to_next'] = np.nan

    return df2

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
    #counts, bin_edges = np.histogram(interval, bins=100, range=(0, 12), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # x軸の大きい値を重視してべき乗則のフィットを行う
    mask = counts > 0
    weights = 1 / bin_centers[mask] ** 2

    # 重み付きフィットを実行
    try:
        initial_guess = [11.95740079,  1.03682929]
        params, params_covariance = curve_fit(func, bin_centers[mask], counts[mask], sigma=weights,  maxfev=3000, p0=initial_guess)
    except RuntimeError as err:
        print("Optimal parameters not found. Using the last tried parameters.")
        params = [11.95740079,  1.03682929]    # 仮の値を設定

    # フィット結果をプロット
    ax.bar(bin_centers, counts, width=np.diff(bin_edges), label='Data')
    ax.plot(bin_centers, func(bin_centers, *params) + 10e-4, label='Fit: a=%.2f, b=%.2f' % tuple(params), color='red')
    ax.set_xlabel( "Interval of Use for \n Recycling Station [h]" )
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(min(bin_centers), max(bin_centers))

    return counts, params, bin_edges, bin_centers

# べき乗則関数を定義
def power_law(x, a, b):
    """
    return : a * np.power(x, b)
    """
    return a * np.power(x, b) +10e-4

# 指数関数を定義
def exp_func(x, a, b):
    """
    return : a * np.exp(-b*x)
    """
    return a*np.exp(-b*x)
    #return a*np.exp(-b*x)+10e-4

def chi_squared_statistic(func, params, bin_centers, counts):
    """
    カイ二乗検定のp値を計算する関数
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

    # log変換
    #counts =   np.log(counts + 10-5)
    #expected = np.log(expected+ 10-5)

    index = 1  # 利用間隔が短いものは重要でないため、解析対象から省く
    
    # カイ二乗適合度検定
    n = len(counts[index:]) # 自由度
    chi_squared_stat, p_value = stats.chisquare(counts[index:]/ np.sum(counts[index:])*n, f_exp=expected[index:] / sum(expected[index:])*n)

    return p_value

def KS_statistic(func, params, bin_centers, counts):
    """
    KS検定のp値を計算する関数
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

    index = 2   # 利用間隔が短いものは重要でないため、解析対象から省く 
    
    # log変換
    counts =   np.log(counts + 0.01)
    expected = np.log(expected+ 0.01)
       
    ks_statistic, p_value = stats.ks_2samp(counts[index:]/ np.sum(counts[index:]), expected[index:] / sum(expected[index:]))    #　KS検定

    return p_value

if __name__ == '__main__':
    print(datetime.datetime.now())
    df = pd.read_csv('data/input/point_history_weather.csv', encoding="utf-8")    

    # 前処理
    df = set_dtype(df)
    df = replace_nan(df)
    df['super'] = df['super'].fillna('イトーヨーカドー')
    df['super'] = df['super'].str.replace(r'\s+', '', regex=True)
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

        fig, ax = plt.subplots(figsize=(4, 4))
        df['interval'] = calc_recycle_period(df, super_name, shop_name)
        counts, params, bin_edges, bin_centers = plot_recycle_period(df['interval'], super_name, shop_name, ax, exp_func)
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

    print(datetime.datetime.now())