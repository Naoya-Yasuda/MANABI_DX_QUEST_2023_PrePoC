import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
from dateutil.relativedelta import relativedelta
import random

# 親ディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 自作モジュール
from utils.point_history_utils import open_point_history_per_shop, aggregate_date, replace_nan, set_dtype, add_date_features
from RS_filliing_rate.RS_fillingrate_test import plot_recycle_period, chi_squared_statistic, exp_func, power_law, KS_statistic, calc_recycle_period

plt.rcParams['font.family'] = 'Meiryo'

# メイン処理
if __name__ == "__main__":
    SEED = 0
    np.random.seed(SEED)
    random.seed(SEED)

    df = pd.read_csv('data/input/point_history_per_shop_date.csv', encoding='utf-8')
    columns_to_drop = ['shop_id', 'shop_name', 'shop_id_1', 'リサイクル分類ID', '支店ID', 'store_opening_time',
                       'store_closing_time', 'rps_opening_time', 'rps_closing_time', '年月日', 'interval_compared_to_next',
                       'amount', 'amount_kg', 'point', 'total_point', 'total_amount', 'coin', 'interval_compared_to_previous',
                       'total_amount_kg_per_day', 'store_latitude', 'store_longitude', '合計全天日射量(MJ/㎡)', '降雪量合計(cm)',
                       '降水量の合計(mm)', '日照時間(時間)']
    df.drop(columns_to_drop, axis=1, inplace=True)

    df_