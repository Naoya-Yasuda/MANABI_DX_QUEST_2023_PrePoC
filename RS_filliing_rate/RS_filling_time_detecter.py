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

# 自作モジュール
from utils.point_history_utils import replace_nan, set_dtype, parse_date
from RS_fillingrate_test import plot_recycle_period, chi_squared_statistic, exp_func, power_law, KS_statistic, calc_recycle_period

# 浮動小数点数を小数点以下2桁で表示するように設定
# pd.set_option('display.float_format', '{:.2f}'.format)

# Windows MatplotlibのデフォルトフォントをMeiryoに設定
plt.rcParams['font.family'] = 'Meiryo'

def plot_interval_per_hour(df, super, shop_name_1, fig_title=None):
    fig, axes = plt.subplots(3,2, figsize=(10,10))
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=16)
    else:
        fig.suptitle(f'{super} {shop_name_1}', fontsize=16)

    # df['interval']が1より大きい行のdf['use_date']について、時刻をhistgramで表示
    bins = np.arange(9, 22, 1)
    ax = axes[0,0]
    #ax.hist(df['use_date'].dt.hour, bins = bins, density=True, color='blue', alpha=0.3, label='all')
    ax.hist(df[df['interval_compared_to_previous'] < 0.3]['use_date'].dt.hour, bins = bins, color='blue', alpha=0.3, label='interval < 0.3')
    ax.set_xlabel('hour')
    ax.set_ylabel('Count (interval < 0.3)')
    ax.legend(loc='upper right')
    ax2 = ax.twinx() 
    ax2.hist(df[df['interval_compared_to_previous'] > 1]['use_date'].dt.hour, bins = bins, color='red', alpha=0.3, label='interval > 1')
    ax2.set_ylabel('Count (interval > 1)')
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    ax.set_title('利用時間帯の分布（前回利用からの経過時間）')
    
    ax = axes[0,1]
    ax.hist(df[df['interval_compared_to_next'] < 0.3]['use_date'].dt.hour, bins = bins, color='blue', alpha=0.3, label='interval < 0.3')
    ax.set_xlabel('hour')
    ax.set_ylabel('Count (interval < 0.3)')
    ax.legend(loc='upper right')
    ax2 = ax.twinx() 
    ax2.hist(df[df['interval_compared_to_next'] > 1]['use_date'].dt.hour, bins = bins, color='red', alpha=0.3, label='interval > 1')
    ax2.set_ylabel('Count (interval > 1)')
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    ax.set_title('利用時間帯の分布（次回利用までの経過時間）')

    # df['use_date'].dtの何時かによってdf['amount_kg']を合計し、棒グラフで表示
    df_temp = df.groupby(df['use_date'].dt.hour)['amount_kg'].sum()
    axes[1,0].bar(df_temp.index, df_temp.values)
    axes[1,0].set_xticks(df_temp.index)
    axes[1,0].set_xlabel('hour')
    axes[1,0].set_ylabel('total recycle amount[kg]')
    axes[1,0].set_title(f'total recycle amount[kg]: {df_temp.sum():.2f}')

    # df['use_date'].dtの何時かによってdf['amount_kg']を平均し、棒グラフで表示
    df_temp = df.groupby(df['use_date'].dt.hour)['amount_kg'].mean()
    axes[2,0].bar(df_temp.index, df_temp.values)
    axes[2,0].set_xticks(df_temp.index)
    axes[2,0].set_xlabel('hour')
    axes[2,0].set_ylabel('mean recycle amount[kg]')
    axes[2,0].set_title(f'mean recycle amount[kg]: {df_temp.mean():.2f}')

    # リサイクルステーション利用間隔のヒストグラムを表示
    counts, params, bin_edges, bin_centers = plot_recycle_period(df['interval_compared_to_next'],super, shop_name_1, axes[2,1], exp_func)
    p_value = chi_squared_statistic(exp_func, params, bin_centers, counts)
    axes[2,1].set_title(f'p-value: {p_value:.5f}') 

    # 1日の総リサイクル量のヒストグラムを表示 (本当にRSの最大搭載量が2tなのか？)
    df_temp = df.groupby(df['use_date'].dt.date)['amount_kg'].sum()
    axes[1,1].hist(df_temp.values, bins=20)
    axes[1,1].set_xlabel('total recycle amount[kg]')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title(f'total recycle amount[kg] histogram per day')
    axes[1,1].set_yscale('log')
    
    # 各グラフの間隔をあける
    fig.tight_layout()
    plt.show()

def open_file(super, shop_name_1): 
    print(f'{super} {shop_name_1}')
    df = pd.read_csv(f'data/input/shop_data/point_history_{super}_{shop_name_1}.csv', encoding="utf-8")
    df = set_dtype(df)
    df = replace_nan(df)
    df = calc_recycle_period(df, super, shop_name_1)
    #df['interval'] = calc_recycle_period(df, "みやぎ生協 ", shop_name_1)
    return df

def plot_recycle_period_per_kg(df1,df2, kg_threshold_less, kg_threshold_more, ax):
    """
    リサイクルステーション利用間隔のkg_threshold_less kg以下とkg_threshold_more kg以上のヒストグラムを比較
    args:
        df1: kg_threshold_less kg以下のデータ
        df2: kg_threshold_more kg以上のデータ
        kg_threshold_less: kg_threshold_less kg以下のデータのヒストグラムを表示
        kg_threshold_more: kg_threshold_more kg以上のデータのヒストグラムを表示
        ax: グラフを描画するaxes
    """
    index = 0
    counts_less, bin_edges_less = np.histogram(df1["interval_compared_to_previous"], bins=100, range=(0, 12), density=True)
    bin_centers_less = (bin_edges_less[:-1] + bin_edges_less[1:]) / 2
    ax.bar(bin_centers_less[index:], counts_less[index:], width=np.diff(bin_edges_less[index:]), alpha=0.5, color='blue',label=f'less than {kg_threshold_less}')

    counts_more, bin_edges_more = np.histogram(df2["interval_compared_to_previous"], bins=100, range=(0, 12), density=True)
    bin_centers_more = (bin_edges_more[:-1] + bin_edges_more[1:]) / 2
    ax.bar(bin_centers_more[index:], counts_more[index:], width=np.diff(bin_edges_more[index:]), alpha=0.5, color='red',label=f'more than {kg_threshold_more}')

    # x軸の大きい値を重視してフィットを行う
    mask = counts_less > 0
    weights = 1 / bin_centers_less[mask] ** 2
    # 重み付きフィットを実行
    try:
        initial_guess = [2.28651235, 3.52252185]
        params_less, params_covariance = curve_fit(exp_func, bin_centers_less[mask], counts_less[mask], sigma=weights,  maxfev=3000, p0=initial_guess)
    except RuntimeError as err:
        print("Optimal parameters not found. Using the last tried parameters.")
        params_less = [2.28651235, 3.52252185]    # 仮の値を設定

    mask[exp_func(bin_centers_less, *params_less) < 0.01] = 0
    weights = 1 / bin_centers_less[mask] ** 2
    param_bounds = ([params_less[0], params_less[1]], [np.inf, np.inf])
    params_more, params_covariance = curve_fit(exp_func, bin_centers_more[mask], counts_more[mask], sigma=weights,  maxfev=3000, bounds=param_bounds)

    ax.plot(bin_centers_less, exp_func(bin_centers_less, *params_less)+10e-6, label=f'Fit less than {kg_threshold_less}', color='blue',alpha=0.5)
    ax.plot(bin_centers_more, exp_func(bin_centers_more, *params_more)+10e-6, label=f'Fit more than {kg_threshold_more}', color='red', alpha=0.5)

    """
    想定される分布と乖離が大きい利用時間において、充填率100%とする
    乖離の量を表す5という数字はマジックナンバー
    """
    fill_index = np.abs(np.log(bin_centers_more+10e-6)-np.log(exp_func(bin_centers_more, *params_more)+10e-6)) > 5  
    fill_index = np.argmax(fill_index)  # fill_indexのTrueの箇所の最小インデックスを取得
    ax.bar(bin_centers_more[fill_index:], counts_more[fill_index:], width=np.diff(bin_edges_more[fill_index:]), alpha=0.5, color='yellow',label=f'filling rate = 100%')

    ax.set_xlabel( "Interval of Use for \n Recycling Station [h]" )
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper right')

if __name__ == '__main__':
    #super = "みやぎ生協"
    #shop_name_1 = "加賀野店"
    super = "ヨークベニマル"
    #shop_name_1 = "若柳店"
    #shop_name_1 = "塩釜店"
    #shop_name_1 = "坂東店"
    #shop_name_1 = "大田原店"
    shop_name_1 = "南中山店"

    #plot_interval_per_hour(super, shop_name_1)
    i = 0
    df_shop_list = pd.read_csv('data/input/shop_list.csv', encoding="utf-8")
    for super, shop_name_1 in zip(df_shop_list['super'], df_shop_list['shop_name_1']):
        #i += 1
        #if i < 10:
        #    continue
        df = open_file(super, shop_name_1)
        # plot_interval_per_hour(df, super, shop_name_1, fig_title=f'{super} {shop_name_1}')

        # # 1日の総リサイクル量が500kg以下の日と1t以上の日を比較
        kg_threshold_less = 500
        kg_threshold_more = 1500
        df_less = df.groupby(df['use_date'].dt.date)['amount_kg'].sum() 
        df_less = df_less[df_less <= kg_threshold_less]
        day_list_less = df_less.index
        df_less = df[df['use_date'].dt.date.isin(day_list_less)]
        df_more = df.groupby(df['use_date'].dt.date)['amount_kg'].sum()
        df_more = df_more[df_more >= kg_threshold_more]
        day_list_more = df_more.index
        df_more = df[df['use_date'].dt.date.isin(day_list_more)]

        if len(df_more) < 10:
            continue

        fig, ax = plt.subplots()
        plot_recycle_period_per_kg(df_less, df_more, kg_threshold_less, kg_threshold_more, ax)
        ax.set_title(f'{super} {shop_name_1}')
        plt.show()

        
        # # # 1日の総リサイクル量が500kg以下の日のみ抽出
        # kg_threshold = 500
        # df_temp = df.groupby(df['use_date'].dt.date)['amount_kg'].sum()
        # df_temp = df_temp[df_temp <= kg_threshold]
        # day_list = df_temp.index
        # df2 = df[df['use_date'].dt.date.isin(day_list)]
        # plot_interval_per_hour(df2, super, shop_name_1, fig_title=f'{super} {shop_name_1} (Total recycle amount (per day) < {kg_threshold} kg)')

        
        # # # 1日の総リサイクル量が1.5t以上の日のみ抽出
        # kg_threshold = 1300
        # df_temp = df.groupby(df['use_date'].dt.date)['amount_kg'].sum()
        # df_temp = df_temp[df_temp >= kg_threshold]
        # day_list = df_temp.index
        # df2 = df[df['use_date'].dt.date.isin(day_list)]
        # if len(df_temp) < 10:
        #     continue
        # plot_interval_per_hour(df2, super, shop_name_1, fig_title=f'{super} {shop_name_1} (Total recycle amount (per day) > {kg_threshold} kg)')





