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
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix

# 親ディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 自作モジュール
from utils.point_history_utils import replace_nan, set_dtype, add_date_features
from RS_filliing_rate.RS_fillingrate_predict import split_data, train_lightgbm, plot_feature_importance

plt.rcParams['font.family'] = 'Meiryo'


def evaluate_model_binary(model,  y_test, y_pred):
    # 正確度（Accuracy）
    accuracy = accuracy_score(y_test, y_pred)

    # 適合率（Precision）
    precision = precision_score(y_test, y_pred)

    # 再現率（Recall）
    recall = recall_score(y_test, y_pred)

    # F1スコア
    f1 = f1_score(y_test, y_pred)

    # 混同行列（Confusion Matrix）
    conf_matrix = confusion_matrix(y_test, y_pred)

    # ROC曲線とAUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    # 2つのクラスが存在するか確認
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_pred)
    else:
        print("Only one class present in y_true. ROC AUC score is not defined in that case.")
        auc = 0

    return accuracy, precision, recall, f1, conf_matrix, auc

# y_test, y_predの混合行列を作成
def plot_confusion_matrix(cm):
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


# メイン処理
if __name__ == "__main__":
    SEED = 0
    np.random.seed(SEED)
    random.seed(SEED)

    aggregated_df = pd.read_csv('data/input/point_history_per_shop_date.csv', encoding='utf-8')
    aggregated_df = set_dtype(aggregated_df)
    
    df_twice_collection_day = pd.read_csv('data/references/RS_twice_collection_day.csv', encoding='utf-8')
    df_twice_collection_day["年月日"] = pd.to_datetime(df_twice_collection_day["年月日"])
    df_twice_collection_day.fillna(0,inplace=True)

    df = pd.merge(aggregated_df, df_twice_collection_day, on=["super", "shop_name_1", "年月日"], how="left")    
    df = set_dtype(df)
    df = replace_nan(df)
    df = add_date_features(df)
    # df['twice_collected_flag']がnanの行を削除
    df = df.dropna(subset=['twice_collected_flag'])
    # df.loc[df['twice_collected_flag'].isna(), 'twice_collected_flag'] = 0
    columns_to_drop = ['shop_id', 'shop_name', 'shop_id_1', 'リサイクル分類ID', '支店ID', 'store_opening_time',
                       'store_closing_time', 'rps_opening_time', 'rps_closing_time',
                       'store_latitude', 'store_longitude','年月日', 'filling_rate', 'amount', ]
    df.drop(columns_to_drop, axis=1, inplace=True)

    categorical_features = ['prefectures', 'municipality', 'shop_name_1', 'super', '天気', 'day_of_week']
    df = pd.get_dummies(df, columns=categorical_features)

    X_train, X_test, y_train, y_test = split_data(df, 'twice_collected_flag')

    # 正のサンプルと負のサンプルの比率を計算
    neg_pos_ratio = y_train[y_train == 0].shape[0] / y_train[y_train == 1].shape[0]

    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'seed': SEED,
        'early_stopping_rounds': 1000,
        'num_iterations': 5000,
        'learning_rate': 0.01,
        'num_leaves': 64,
        #'scale_pos_weight': neg_pos_ratio
    }
    model = train_lightgbm(X_train, y_train, X_test, lgb_params)
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)  # 0.5を閾値として二値化
    
    accuracy, precision, recall, f1, conf_matrix, auc = evaluate_model_binary(model, y_test, y_pred)
    
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'AUC: {auc}')

    plot_confusion_matrix(conf_matrix)
    plot_feature_importance(model, X_train, 20)






    # aggregated_df_once_collected =  aggregated_df[aggregated_df["twice_collected_flag"] == 0]
    # aggregated_df_twice_collected = aggregated_df[aggregated_df["twice_collected_flag"] == 1]

    # df_twice_collection_day = pd.read_csv('data/references/RS_twice_collection_day.csv', encoding='utf-8')
    # df_twice_collection_day["年月日"] = pd.to_datetime(df_twice_collection_day["年月日"])




    # df_shop_list = pd.read_csv('data/input/shop_list.csv', encoding='utf-8')
    # num_shop = len(df_shop_list)

    # for i in tqdm(range(num_shop)):
    #     super_name = df_shop_list['super'][i]
    #     shop_name = df_shop_list['shop_name_1'][i]
        
    #     print(f'{super_name} {shop_name} is processing...')
    #     df = open_point_history_per_shop(super_name, shop_name)
    #     df = pd.merge(df, df_twice_collection_day, on=["super", "shop_name_1", "年月日"], how="left")
    #     df.fillna(0,inplace=True)
    #     df_once_collected = df[df["twice_collected_flag"] == 0]
    #     df_twice_collected = df[df["twice_collected_flag"] == 1]

    #     kg_threshold = aggregated_df_twice_collected[(aggregated_df_twice_collected['super'] == super_name) & \
    #                                                  (aggregated_df_twice_collected['shop_name_1'] == shop_name)]["amount_kg"].mean()
    #     print(kg_threshold)
    #     if kg_threshold is np.nan:
    #         continue
    #     df_once_collected = extract_high_recycling_days(df_once_collected, 1300)
    #     if (df_twice_collected.size < 1000) or (df_once_collected.size <1000):
    #         continue

    #     fig, axes = plt.subplots(2,1)
    #     # interval_compared_to_previous,interval_compared_to_next
    #     # plot_recycle_period(df['interval_compared_to_previous'], super_name, shop_name, axes[0], exp_func)
    #     counts, params, bin_edges, bin_centers = plot_recycle_period(df_once_collected['interval_compared_to_previous'], super_name, shop_name, axes[0], exp_func)
    #     counts, params, bin_edges, bin_centers = plot_recycle_period(df_twice_collected['interval_compared_to_previous'], super_name, shop_name, axes[1], exp_func)
    #     plt.show()

        # メモ
        # 最終利用時間から充填率100%を判定できないか




    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # axes[0].set_title("amount_kg")
    # axes[0].hist(df_once_collected["amount_kg"], bins=20,  color="blue", alpha=0.5, label="once_collected" , density=True)
    # axes[0].hist(df_twice_collected["amount_kg"], bins=20, color="red", alpha=0.5 , label="twice_collected", density=True)
    # axes[1].set_title("filling_rate")
    # axes[1].hist(df_once_collected["filling_rate"], bins=20,  color="blue", alpha=0.5, label="once_collected", density=True)
    # axes[1].hist(df_twice_collected["filling_rate"], bins=20, color="red", alpha=0.5, label="twice_collected", density=True)
    # plt.show()





    # print(df["平均雲量(10分比)"].value_counts(dropna=False))

    # aggregation = {
    #     'amount': 'mean',
    #     'amount_kg': 'mean',
    #     'point': 'mean',
    #     'total_point': 'mean',
    #     'total_amount': 'mean',
    #     'coin': 'mean',
    #     'interval_compared_to_previous': 'max',
    #     'interval_compared_to_next': 'max',
    # }
    
    # df2 = df.groupby(['twice_collected_flag']).agg(aggregation)

    # print(df2.to_markdown())

    



    