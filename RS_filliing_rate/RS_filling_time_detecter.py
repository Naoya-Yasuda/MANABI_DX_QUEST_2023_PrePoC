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
    df_shop_list = pd.read_csv('data/input/shop_list.csv', encoding="utf-8")
    for super, shop_name_1 in zip(df_shop_list['super'], df_shop_list['shop_name_1']):
        df = open_file(super, shop_name_1)    
        if 'interval' in df.columns:
            df = df.drop(columns=['interval'])
        df_temp = df.groupby(df['use_date'].dt.date)['amount_kg'].sum()
        df_temp = df_temp[df_temp >= 1500]
        if len(df_temp) < 10:
            continue


        plot_interval_per_hour(df, super, shop_name_1, fig_title=f'{super} {shop_name_1}')
        
        # # 1日の総リサイクル量が1t以下の日のみ抽出
        kg_threshold = 500
        df_temp = df.groupby(df['use_date'].dt.date)['amount_kg'].sum()
        df_temp = df_temp[df_temp <= kg_threshold]
        day_list = df_temp.index
        df2 = df[df['use_date'].dt.date.isin(day_list)]
        plot_interval_per_hour(df2, super, shop_name_1, fig_title=f'{super} {shop_name_1} (Total recycle amount (per day) < {kg_threshold} kg)')

        
        # # 1日の総リサイクル量が2t以上の日とその前後1日のみ抽出
        kg_threshold = 1500
        df_temp = df.groupby(df['use_date'].dt.date)['amount_kg'].sum()
        df_temp = df_temp[df_temp >= kg_threshold]
        day_list = df_temp.index
        df2 = df[df['use_date'].dt.date.isin(day_list)]
        if len(df_temp) < 10:
            continue
        plot_interval_per_hour(df2, super, shop_name_1, fig_title=f'{super} {shop_name_1} (Total recycle amount (per day) > {kg_threshold} kg)')





