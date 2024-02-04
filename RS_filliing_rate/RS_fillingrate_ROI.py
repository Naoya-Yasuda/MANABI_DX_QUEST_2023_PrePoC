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
import pickle
import glob
import re

# 親ディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 自作モジュール
from utils.point_history_utils import open_point_history_per_shop, aggregate_date, replace_nan, set_dtype, add_date_features, set_previous_data
from RS_filliing_rate.RS_fillingrate_test import plot_recycle_period, chi_squared_statistic, exp_func, power_law, KS_statistic, calc_recycle_period
from RS_filliing_rate.RS_fillingrate_predict import preprocessing, split_data, train_lightgbm, evaluate_model, plot_results
from RS_filliing_rate.visualization_twice_collection_day import evaluate_model_binary, plot_confusion_matrix

plt.rcParams['font.family'] = 'Meiryo'


def load_model(file_pattern='models/RS_fillingrate_lightgbm_rmse_*.pkl'):
    # 指定されたパターンに一致するファイルのリストを取得
    file_list = glob.glob(file_pattern)

    # ファイル名から数字を抽出し、数値に変換してリストに格納
    file_numbers = []
    for file in file_list:
        match = re.search(r'(\d+)\.pkl$', file)
        if match:
            file_numbers.append((int(match.group(1)), file))

    # 数値が最小のファイルを特定
    if file_numbers:
        min_number_file = min(file_numbers, key=lambda x: x[0])[1]

        # 最小の数字を持つファイルを読み込み
        with open(min_number_file, 'rb') as f:
            model = pickle.load(f)
    else:
        print("No files found matching the pattern.")
        model = None
    return model


def detect_best_auc_param_1(model, y_test, y_pred):
    # 最適なスコアとその閾値を保存する変数を初期化
    best_auc = 0
    best_thresholds = (0, 0)

    # y_pred_threshold と y_test_threshold の範囲とステップサイズを定義
    pred_threshold_range = np.arange(0.5, 1.01, 0.01)
    test_threshold_range = np.arange(0.99, 1.01, 0.01)

    # すべての組み合わせを試す
    for y_pred_threshold in pred_threshold_range:
        for y_test_threshold in test_threshold_range:
            # 二値予測と実際の値を閾値で変換
            y_pred_bin = y_pred >= y_pred_threshold
            y_test_bin = y_test >= y_test_threshold
            
            # モデルを評価
            accuracy, precision, recall, f1, conf_matrix, auc = evaluate_model_binary(model, y_test_bin, y_pred_bin)
            
            # aucスコアが現在の最良のスコアよりも良い場合、更新
            if auc > best_auc:
                best_auc = auc
                best_thresholds = (y_pred_threshold, y_test_threshold)

    # 最良の結果を出力
    print(f'Best auc Score: {best_auc}')
    print(f'Best Thresholds - y_pred_threshold: {best_thresholds[0]}, y_test_threshold: {best_thresholds[1]}')


def detect_best_auc_param_2(model, y_test, y_pred):
    # 最適なスコアとその閾値を保存する変数を初期化
    best_auc = 0
    best_thresholds = (0, 0)

    # y_pred_threshold と y_test_threshold の範囲とステップサイズを定義
    pred_threshold_range = np.arange(0.25, 0.41, 0.01)
    test_threshold_range = np.arange(0.25, 0.41, 0.01)

    # すべての組み合わせを試す
    for y_pred_threshold in pred_threshold_range:
        for y_test_threshold in test_threshold_range:
            # 二値予測と実際の値を閾値で変換
            y_pred_bin = y_pred <= y_pred_threshold
            y_test_bin = y_test<= y_test_threshold
            
            # モデルを評価
            accuracy, precision, recall, f1, conf_matrix, auc = evaluate_model_binary(model, y_test_bin, y_pred_bin)
            
            # aucスコアが現在の最良のスコアよりも良い場合、更新
            if auc > best_auc:
                best_auc = auc
                best_thresholds = (y_pred_threshold, y_test_threshold)

    # 最良の結果を出力
    print(f'Best auc Score: {best_auc}')
    print(f'Best Thresholds - y_pred_threshold: {best_thresholds[0]}, y_test_threshold: {best_thresholds[1]}')

# 回収を増やした場合の追加古紙量を算出
# 最終利用時間とRPS時間の差　×　その日の投入量／時間
def calc_ROI_add_collection(df, y_test, y_pred, test_size=0.2):
    y_pred_bin = y_pred >= 0.5
    y_test_bin = y_test >= 1
    df_full = df[y_pred_bin]
    df_full['full_duration'] = np.where((df_full['rps_closing_time'] - df_full['use_date']).dt.total_seconds() > 0,\
                                         (df_full['rps_closing_time'] - df_full['use_date']).dt.total_seconds(), 0) / 3600
    df_full['add_collection'] = df_full['full_duration'] * df_full['amount_kg']\
                                 / (df_full['use_date'] - df_full['rps_opening_time']).dt.total_seconds() * 3600
    total_add_collection_per_year = df_full.loc[df_full['年月日'].dt.year == 2023,'add_collection'].sum() / test_size   # ここでは最もデータがそろっている2023を抜き出し
    return total_add_collection_per_year

if __name__ == "__main__":
    SEED = 0
    np.random.seed(SEED)
    random.seed(SEED)
    test_size = 0.2
    used_paper_price_per_kg = 10.5


    file_pattern = 'models/RS_fillingrate_lightgbm_rmse_*.pkl'
    model = load_model(file_pattern)
    df = pd.read_csv('data/input/point_history_per_shop_date.csv', encoding='utf-8')
    df = preprocessing(df)
    columns_to_drop = ['shop_id', 'shop_name', 'shop_id_1', 'リサイクル分類ID', '支店ID', 'store_opening_time',
                       'store_closing_time', 'rps_opening_time', 'rps_closing_time', '年月日', 'interval_compared_to_next',
                       'amount', 'amount_kg', 'point', 'total_point', 'total_amount', 'coin', 'interval_compared_to_previous',
                       'store_latitude', 'store_longitude', '合計全天日射量(MJ/㎡)', '降雪量合計(cm)',
                       '降水量の合計(mm)', '日照時間(時間)', 'use_date', 'total_amount_kg_per_day',]
    df_drop = df.drop(columns_to_drop, axis=1, errors='ignore')

    categorical_features = ['prefectures', 'municipality', 'shop_name_1', 'super', '天気', 'day_of_week']
    df_drop = pd.get_dummies(df_drop, columns=categorical_features)

    X_train, X_test, y_train, y_test, indices_train, indices_test = split_data(df_drop, 'filling_rate', test_size=test_size, random_state=SEED)
    y_pred = model.predict(X_test)
    total_add_collection_per_year = calc_ROI_add_collection(df.loc[indices_test], y_test, y_pred, test_size)
    print('古紙回収量(年あたり)', df.loc[df['年月日'].dt.year == 2023,'amount_kg'].sum(), 'kg/year')
    print(f'充填率100%による逸失古紙回収量 {int(total_add_collection_per_year)} kg/year')
    print('充填率100%による逸失利益（年あたり）', int(total_add_collection_per_year * used_paper_price_per_kg), '円/year')
    print('充填率100%による逸失利益（月あたり）', int(total_add_collection_per_year * used_paper_price_per_kg / 12), '円/month')
    print('充填率100%による逸失利益（月あたり）', int(total_add_collection_per_year * used_paper_price_per_kg / 12), '円/month')






    # accuracy, precision, recall, f1, conf_matrix, auc = evaluate_model_binary(model, y_test_bin, y_pred_bin)
    
    # print(f'Accuracy: {accuracy}')
    # print(f'Precision: {precision}')
    # print(f'Recall: {recall}')
    # print(f'F1 Score: {f1}')
    # print(f'Confusion Matrix:\n{conf_matrix}')
    # print(f'AUC: {auc}')

    # plot_confusion_matrix(conf_matrix)





    # y_pred_bin = y_pred <= 0.3
    # y_test_bin = y_test <= 0.3
    # accuracy, precision, recall, f1, conf_matrix, auc = evaluate_model_binary(model, y_test_bin, y_pred_bin)
    
    # print(f'Accuracy: {accuracy}')
    # print(f'Precision: {precision}')
    # print(f'Recall: {recall}')
    # print(f'F1 Score: {f1}')
    # print(f'Confusion Matrix:\n{conf_matrix}')
    # print(f'AUC: {auc}')

    # plot_confusion_matrix(conf_matrix)










    # メモ
    # # f1が最良
    # y_pred_bin = y_pred >= 0.68
    # y_test_bin = y_test >= 0.9


 
    # # 100%になる日（予測値）を算出
    # y_pred = model.predict(X_test)
    # # y_predが0.95以上のdfを作成
    # df_predict_full = df.loc[y_test[y_pred >= 0.95].index]
    

    