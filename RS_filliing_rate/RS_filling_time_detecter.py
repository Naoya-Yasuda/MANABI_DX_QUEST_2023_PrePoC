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
from utils.point_history_utils import open_point_history_per_shop, aggregate_date
from RS_filliing_rate.RS_fillingrate_test import plot_recycle_period, chi_squared_statistic, exp_func, power_law, KS_statistic, calc_recycle_period

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

def extract_low_recycling_days(df, kg_threshold):
    """
    1日のリサイクル量がkg_threshold未満の日のデータ行のみ抽出する関数
    args:
        df: pandas.DataFrame
        kg_threshold: float
    return:
        df_low: pandas.DataFrame
    """
    df_low = df.groupby(df['use_date'].dt.date)['amount_kg'].sum() 
    df_low = df_low[df_low < kg_threshold]
    day_list_low = df_low.index
    df_low = df[df['use_date'].dt.date.isin(day_list_low)]
    return df_low

def extract_high_recycling_days(df, kg_threshold):
    """
    1日のリサイクル量がkg_thresholdより大きい日のデータのみ抽出する関数
    args:
        df: pandas.DataFrame
        kg_threshold: float
    return:
        df_high: pandas.DataFrame
    """
    df_high = df.groupby(df['use_date'].dt.date)['amount_kg'].sum()
    df_high = df_high[df_high >= kg_threshold]
    day_list_high = df_high.index
    df_high = df[df['use_date'].dt.date.isin(day_list_high)]
    return df_high

def calc_maxfilling_hour(df_low,df_high, option_graph_obj_return=False):
    """
    リサイクルステーションの充填率が100%になる時間を計算する関数
    args:
        df_low: リサイクルステーション利用間隔が短い日のみ抽出したデータフレーム
        df_high: リサイクルステーション利用間隔が長い日のみ抽出したデータフレーム
        option_graph_obj_return: グラフのオブジェクトを返すかどうか
    return（option_graph_obj_returnがTrueの場合）:
        max_filling_hour: リサイクルステーションの充填率が100%になる時間 （もしなければnp.nanを返す）
    return (option_graph_obj_returnがFalseの場合):
        max_filling_hour: リサイクルステーションの充填率が100%になる時間 （もしなければnp.nanを返す）
        counts_low: リサイクルステーション利用間隔が短い日のヒストグラムの度数
        bin_edges_low: リサイクルステーション利用間隔が短い日のヒストグラムのビンの境界
        bin_centers_low: リサイクルステーション利用間隔が短い日のヒストグラムのビンの中心
        counts_high: リサイクルステーション利用間隔が長い日のヒストグラムの度数
        bin_edges_high: リサイクルステーション利用間隔が長い日のヒストグラムのビンの境界
        bin_centers_high: リサイクルステーション利用間隔が長い日のヒストグラムのビンの中心
        params_low: リサイクルステーション利用間隔が短い日のフィット関数のパラメータ
        params_high: リサイクルステーション利用間隔が長い日のフィット関数のパラメータ
        fill_index: 充填率100%とするインデックス
    """
    counts_low, bin_edges_low = np.histogram(df_low["interval_compared_to_previous"], bins=100, range=(0, 12), density=True)
    bin_centers_low = (bin_edges_low[:-1] + bin_edges_low[1:]) / 2

    counts_high, bin_edges_high = np.histogram(df_high["interval_compared_to_previous"], bins=100, range=(0, 12), density=True)
    bin_centers_high = (bin_edges_high[:-1] + bin_edges_high[1:]) / 2

    # x軸の大きい値を重視してフィットを行う
    mask = counts_low > 0
    weights = 1 / bin_centers_low[mask] ** 2
    # 重み付きフィットを実行
    try:
        initial_guess = [2.28651235, 3.52252185]
        params_low, params_covariance = curve_fit(exp_func, bin_centers_low[mask], counts_low[mask], sigma=weights,  maxfev=3000, p0=initial_guess)
    except RuntimeError as err:
        print("Optimal parameters not found. Using the last tried parameters.")
        params_low = [2.28651235, 3.52252185]    # 仮の値を設定

    mask[exp_func(bin_centers_low, *params_low) < 0.01] = 0
    weights = 1 / bin_centers_low[mask] ** 2
    param_bounds = ([params_low[0], params_low[1]], [np.inf, np.inf])
    params_high, params_covariance = curve_fit(exp_func, bin_centers_high[mask], counts_high[mask], sigma=weights,  maxfev=3000, bounds=param_bounds)

    """
    想定される分布と乖離が大きい利用時間において、充填率100%とする
    乖離の量を表す5という数字はマジックナンバー
    """
    fill_index = np.abs(np.log(bin_centers_high+10e-6)-np.log(exp_func(bin_centers_high, *params_high)+10e-6)) > 5  
    fill_index = np.argmax(fill_index)  # fill_indexのTrueの箇所の最小インデックスを取得

    if np.max(counts_high[fill_index:]) == 0:
        max_filling_hour = np.nan
    else:
        max_filling_hour = bin_edges_high[fill_index]

    if option_graph_obj_return:
        return max_filling_hour ,counts_low, bin_edges_low, bin_centers_low, counts_high, bin_edges_high, bin_centers_high, params_low, params_high, fill_index
    else:
        return max_filling_hour

def plot_recycle_period_per_kg(bin_edges_low, bin_centers_low,counts_low, bin_edges_high,bin_centers_high, counts_high, kg_threshold_low, \
                               kg_threshold_high, params_low, params_high, fill_index, ax, super, shop_name_1):
    """
    リサイクルステーション利用間隔のkg_threshold_low kg以下とkg_threshold_high kg以上のヒストグラムを比較
    args:
        bin_edges_low: ビンの境界
        bin_centers_low: ビンの中心
        counts_low: ヒストグラムの度数
        bin_edges_high: ビンの境界
        bin_centers_high: ビンの中心
        counts_high: ヒストグラムの度数
        kg_threshold_low: kg_threshold_low kg以下の日のみ抽出
        kg_threshold_high: kg_threshold_high kg以上の日のみ抽出
        params_low: フィット関数のパラメータ
        params_high: フィット関数のパラメータ
        fill_index: 充填率100%とするインデックス
        ax: サブプロット
        super: スーパー名
        shop_name_1: 店舗名
    return:
        None
    """
    
    ax.bar(bin_centers_low, counts_low, width=np.diff(bin_edges_low), alpha=0.5, color='blue',label=f'low than {kg_threshold_low}')
    ax.bar(bin_centers_high, counts_high, width=np.diff(bin_edges_high), alpha=0.5, color='red',label=f'high than {kg_threshold_high}')
    ax.plot(bin_centers_low, exp_func(bin_centers_low, *params_low)+10e-6, label=f'Fit low than {kg_threshold_low}', color='blue',alpha=0.5)
    ax.plot(bin_centers_high, exp_func(bin_centers_high, *params_high)+10e-6, label=f'Fit high than {kg_threshold_high}', color='red', alpha=0.5)
    ax.bar(bin_centers_high[fill_index:], counts_high[fill_index:], width=np.diff(bin_edges_high[fill_index:]), alpha=0.5, color='yellow',label=f'filling rate = 100%')

    ax.set_xlabel( "Interval of Use for \n Recycling Station [h]" )
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(bin_centers_high[0], bin_centers_high[-60])
    ax.legend(loc='upper right')
    ax.set_title(f'{super} {shop_name_1}')
    plt.show()

def calc_filling_rate(df, max_filling_hour, kg_threshold=1300, kg_threshold_2=1700):
    """
    各行の'filling_rate'を計算する。充填率100%とする場合は以下の通り。
    １．1日でkg_threshold 以上リサイクル量がある & 前回のリサイクルから max_filling_hour 以上時間が空いている
    ２．1日でkg_threshold_2 以上リサイクル量がある
    args:
        df: dataframe
        max_filling_hour: float[h]
        kg_threshold: float[kg]
        kg_threshold_2: float[kg]
    return:
        df2: dataframe    
    """
    df_high = extract_high_recycling_days(df, kg_threshold)
    if len(df_high[df_high['interval_compared_to_next'] > max_filling_hour]) > 0:
        df_high.loc[df_high['interval_compared_to_next'] > max_filling_hour, 'filling_rate'] = 1
        # 'filling_rate'が1で、次の行の'use_date'が20時以降　かつ　次の行の'use_date'が同じ日の場合、その行の'filling_rate'を1にする
        for i in df_high.index[:-1]:
            if df_high.loc[i, 'filling_rate'] == 1:
                next_index = df_high.index[df_high.index.get_loc(i) + 1]
                if df_high.loc[i, 'use_date'].day == df_high.loc[next_index, 'use_date'].day and df_high.loc[next_index, 'use_date'].hour >= 20:
                    df_high.loc[next_index, 'filling_rate'] = 1
        df2 = pd.merge(df, df_high[['use_date', 'filling_rate']], on='use_date', how='left')
    else:
        df2 = df.copy()
        df2['filling_rate'] = np.nan
    
    # kg_threshold_2 以上の日は最終行の'filling_rate'を1にする
    df_high = extract_high_recycling_days(df, kg_threshold_2)
    if len(df_high) > 0:
        df_high['filling_rate'] = np.nan
        # 次の行の'use_date'が別の日の場合、その行の'filling_rate'を1にする
        for i in df_high.index[:-1]:
            next_index = df_high.index[df_high.index.get_loc(i) + 1]
            if df_high.loc[i, 'use_date'].day != df_high.loc[next_index, 'use_date'].day:
                df_high.loc[i, 'filling_rate'] = 1
        index_list = df_high[df_high['filling_rate'] == 1].index
        df2.loc[index_list, 'filling_rate'] = 1

    # 各行の'filling_rate'を計算する
    aggregate_df = aggregate_date(df2)
    for date, max_amount_kg, filling_rate in zip(aggregate_df['年月日'], aggregate_df['amount_kg'], aggregate_df['filling_rate']):
        if filling_rate != 1.0:
            max_amount_kg = kg_threshold_2
        total_amount_kg_per_day = 0
        for i in df2[df2['年月日'] == date].index:
            total_amount_kg_per_day += df2.loc[i, 'amount_kg']
            df2.loc[i, 'total_amount_kg_per_day'] = total_amount_kg_per_day
            df2.loc[i, 'filling_rate'] = total_amount_kg_per_day / max_amount_kg
    return df2

if __name__ == '__main__':
    df_shop_list = pd.read_csv('data/input/shop_list.csv', encoding="utf-8")
    # for super, shop_name_1 in zip(df_shop_list['super'], df_shop_list['shop_name_1']):
    #     df = open_point_history_per_shop(super, shop_name_1)

        # # # 1日の総リサイクル量が500kg以下の日と1.3t以上の日を比較
        # kg_threshold_low = 500
        # kg_threshold_high = 1300
        # df_low = extract_low_recycling_days(df, kg_threshold_low)
        # df_high = extract_high_recycling_days(df, kg_threshold_high)

        # if len(df_high) < 10:
        #     print(f'{super} {shop_name_1} is not enough data')
        #     # df_shop_listのsuper, shop_name_1が一致する行で、"max_filling_hour"という新しい列にnp.nanを代入
        #     df_shop_list.loc[(df_shop_list['super'] == super) & (df_shop_list['shop_name_1'] == shop_name_1), 'max_filling_hour'] = np.nan
        #     continue

        # # リサイクルステーションの充填率が100%になる時間を計算
        # try:
        #     max_filling_hour, counts_low, bin_edges_low, bin_centers_low, counts_high, bin_edges_high, bin_centers_high, params_low, params_high, fill_index \
        #         = calc_maxfilling_hour(df_low,df_high, option_graph_obj_return=True)
            
        #     fig, ax = plt.subplots()
        #     plot_recycle_period_per_kg(bin_edges_low, bin_centers_low,counts_low, bin_edges_high,bin_centers_high, counts_high, kg_threshold_low, \
        #                            kg_threshold_high, params_low, params_high, fill_index, ax, super, shop_name_1)
            
        #     print(f'max_filling_hour: {max_filling_hour:.2f}')
        #     # df_shop_listのsuper, shop_name_1が一致する行で、"max_filling_hour"という新しい列にmax_filling_hourを代入
        #     df_shop_list.loc[(df_shop_list['super'] == super) & (df_shop_list['shop_name_1'] == shop_name_1), 'max_filling_hour'] = max_filling_hour
        # except Exception as e:  # Exceptionクラスで全ての例外を全て捕捉してしまう
        #     print(e)
        #     df_shop_list.loc[(df_shop_list['super'] == super) & (df_shop_list['shop_name_1'] == shop_name_1), 'max_filling_hour'] = np.nan

    i = 0
    aggregated_df = pd.DataFrame()
    for super, shop_name_1, max_filling_hour in tqdm(zip(df_shop_list['super'], df_shop_list['shop_name_1'], df_shop_list['max_filling_hour']), total=len(df_shop_list)):
        print(f'{super} {shop_name_1} is processing...')
        df = open_point_history_per_shop(super, shop_name_1)
        df = calc_filling_rate(df, max_filling_hour,kg_threshold=1300, kg_threshold_2=1700)
        df.to_csv(f'data/input/shop_data/point_history_{super}_{shop_name_1}.csv', index=False, encoding="utf-8")
        aggregated_df_temp = aggregate_date(df)
        aggregated_df = pd.concat([aggregated_df, aggregated_df_temp]).reset_index(drop=True)

        
    aggregated_df.to_csv('data/input/point_history_per_shop_date2.csv', index=False, encoding="utf-8")


