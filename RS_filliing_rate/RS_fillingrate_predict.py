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

# 親ディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 自作モジュール
from utils.point_history_utils import replace_nan, set_dtype, add_date_features, set_previous_data

plt.rcParams['font.family'] = 'Meiryo'


def preprocessing(df):
    df = set_dtype(df)
    df = replace_nan(df)
    df = add_date_features(df)
    df.loc[df["filling_rate"] > 1, "filling_rate"] = 1
    df = set_previous_data(df, ['amount_kg', 'filling_rate'], years=1)
    return df

def split_data(df, target_column, test_size=0.2, random_state=0):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    indices = np.array(range(X.shape[0]))
    X_train, X_test, y_train, y_test,indices_train, indices_test = train_test_split(X, y, indices, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, indices_train, indices_test

def train_lightgbm(X_train, y_train, lgb_params):
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(lgb_params, train_data, valid_sets=test_data)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, rmse, r2, y_pred

def plot_results(y_test, y_pred, r2):
    plt.figure(figsize=(5, 4))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.title(f'決定係数: {round(r2, 2)}')
    plt.xlabel('充填率（正解値）')
    plt.ylabel('充填率（予測値）')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, X_train, feature_num=None):
    feature_importances = model.feature_importance(importance_type='split')
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    if feature_num is not None:
        feature_importance_df = feature_importance_df[:feature_num]
    plt.figure(figsize=(5, 100))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('予測における重要度')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

# メイン処理
if __name__ == "__main__":
    SEED = 0
    np.random.seed(SEED)
    random.seed(SEED)
    test_size = 0.2

    df = pd.read_csv('data/input/point_history_per_shop_date.csv', encoding='utf-8')
    df = preprocessing(df)
    columns_to_drop = ['shop_id', 'shop_name', 'shop_id_1', 'リサイクル分類ID', '支店ID', 'store_opening_time',
                       'store_closing_time', 'rps_opening_time', 'rps_closing_time', '年月日', 'interval_compared_to_next',
                       'amount', 'amount_kg', 'point', 'total_point', 'total_amount', 'coin', 'interval_compared_to_previous',
                       'store_latitude', 'store_longitude', '合計全天日射量(MJ/㎡)', '降雪量合計(cm)',
                       '降水量の合計(mm)', '日照時間(時間)', 'use_date', 'total_amount_kg_per_day',]
    df.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

    categorical_features = ['prefectures', 'municipality', 'shop_name_1', 'super', '天気', 'day_of_week']
    df = pd.get_dummies(df, columns=categorical_features)

    X_train, X_test, y_train, y_test, _, _ = split_data(df, 'filling_rate', test_size=test_size, random_state=SEED)

    lgb_params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'seed': 0,
        'early_stopping_rounds' : 1000,
        'num_iterations' : 10000,
        'learning_rate' : 0.02,
        'max_depth': 8,
    }
    model = train_lightgbm(X_train, y_train, lgb_params)

 
    mse, mae, rmse, r2, y_pred = evaluate_model(model, X_test, y_test)

    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Mean Absolute Error  (MAE): {mae}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared (R2): {r2}')

    rmse_truncated = format(rmse, '.4f')
    with open(f'models/RS_fillingrate_lightgbm_rmse_{rmse_truncated}.pkl', 'wb') as f:
        pickle.dump(model, f)
    plot_results(y_test, y_pred, r2)
    plot_feature_importance(model, X_train, 20)
