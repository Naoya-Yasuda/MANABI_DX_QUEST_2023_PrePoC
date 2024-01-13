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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 親ディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 自作モジュール
from utils.point_history_utils import open_point_history_per_shop, aggregate_date, replace_nan, set_dtype
from RS_filliing_rate.RS_fillingrate_test import plot_recycle_period, chi_squared_statistic, exp_func, power_law, KS_statistic, calc_recycle_period

# 日付特徴量の追加
def add_date_features(df):
    df["month"] = df["年月日"].dt.month
    df["day"] = df["年月日"].dt.day
    df["year"] = df["年月日"].dt.year
    return df

df = pd.read_csv('data/input/point_history_per_shop_date.csv', encoding='utf-8')

df = set_dtype(df)
df = replace_nan(df)
df = add_date_features(df)

# Drop unnecessary columns
columns_to_drop = ['series_id', 'shop_id', 'shop_name', 'shop_id_1', 'リサイクル分類ID', '支店ID', 'store_opening_time', 'store_closing_time', 'rps_opening_time', 'rps_closing_time','年月日', 'interval_compared_to_next']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Handle categorical variables with one-hot encoding
categorical_columns = ['prefectures', 'municipality','shop_name_1','super', '天気']
df = pd.get_dummies(df, columns=categorical_columns)

# Split the data into features and target
X = df.drop('filling_rate', axis=1)
y = df['filling_rate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the LightGBM model
train_data = lgb.Dataset(X_train, label=y_train)
param = {'num_leaves': 31, 'objective': 'regression'}
num_round = 100
bst = lgb.train(param, train_data, num_round)

# Predict and evaluate the model
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

# Evaluate the predictions
# Calculate and print evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R2): {r2}')

print("actual")
print(y_test[:10].values)
print("pred")
print(y_pred[:10])



















# df_train_machine = prepare(base_train,actual_train,processing_train)
# #df_train_machine = df_train_machine[df_train_machine['数量1'].isna()==False]
# features_del = ["削除フラグ","削除担当者コード","削除端末コード","取込キー","更新回数","更新担当者コード","更新端末コード","削除日","作成日","作成担当者コード","作成端末コード",\
#                 "製造完成数量","完了区分","完了区分名","更新日"]
# for f in features_del:
#     df_train_machine = df_train_machine.drop(f, axis=1)
# df_train_machine = add_additionalworktime(df_train_machine)

# lgb_params = {
#     'objective': 'regression',
#     'boosting_type': 'gbdt',
#     'seed': 0,
#     'objective':'mae', 
#     'metric':'mae',
#     'early_stopping_rounds' : 1000,
#     'num_iterations' : 10000,
#     'learning_rate' : 0.02,
#     'max_depth': 8,
#     'num_leaves': 16,
#     #'categorical_feature': categorical_features
# }

# for c in categorical_features:
#     df_train_machine[c] = df_train_machine[c].astype('category')
#     df_val_machine[c] = df_val_machine[c].astype('category')
#     df_test_machine[c] = df_test_machine[c].astype('category')

# lgb_train_machine_work = lgb.Dataset(df_train_machine[features], df_train_machine["作業時間"], categorical_feature=categorical_features)
# lgb_val_machine_work = lgb.Dataset(df_val_machine[features], df_val_machine["作業時間"], categorical_feature=categorical_features)

# lgb_train_machine_addwork = lgb.Dataset(df_train_machine[features], df_train_machine["作業付帯時間"], categorical_feature=categorical_features)
# lgb_val_machine_addwork = lgb.Dataset(df_val_machine[features], df_val_machine["作業付帯時間"], categorical_feature=categorical_features)

# model_machine_work = lgb.train(lgb_params, lgb_train_machine_work, valid_sets=lgb_val_machine_work)

# y_test = model_machine_work.predict(df_val_machine[features])