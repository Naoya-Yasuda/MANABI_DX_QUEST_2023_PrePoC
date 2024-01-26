import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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
from utils.point_history_utils import open_point_history_per_shop, aggregate_date, replace_nan, set_dtype, aggregate_shop_date
from RS_filliing_rate.RS_fillingrate_test import plot_recycle_period, chi_squared_statistic, exp_func, power_law, KS_statistic, calc_recycle_period
from RS_filliing_rate.RS_fillingrate_predict import add_date_features, set_previous_data, arrange_df, split_data


plt.rcParams['font.family'] = 'Meiryo'
np.set_printoptions(500)


# LSTMモデルの定義
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size=100, output_size=5):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]  # 最後の5日分の予測を返す
    
    
# データをシーケンスに分割する関数
def create_inout_sequences(input_data, target_data, fill_zero_columns):
    inout_seq = []
    L = len(input_data)
    for i in range(365, L - 5):  # 5日分の余裕を持たせる
        train_seq = input_data[:i]
        #for column_position in fill_zero_columns:
        #    train_seq[-28:, column_position] = 0
        
        #train_seq = np.delete(train_seq, slice(-28, None), axis=0)
        train_label = target_data[i :i + 5]  # 1日後から5日後までのデータ
        inout_seq.append((train_seq, train_label))
    return inout_seq




# メイン処理
if __name__ == "__main__":
    SEED = 0
    np.random.seed(SEED)
    random.seed(SEED)

    df = pd.read_csv('data/input/point_history_per_shop_date.csv', encoding='utf-8')
    
    #df = aggregate_shop_date(df)
    df = arrange_df(df)
    df = add_date_features(df)
    columns_to_drop = ['shop_name', 'shop_id', 'shop_id_1', 'リサイクル分類ID', '支店ID', 'store_opening_time',
                       'store_closing_time', 'rps_opening_time', 'rps_closing_time', '年月日', 'interval_compared_to_next',
                       'amount', 'point', 'total_point', 'total_amount', 'coin', 'interval_compared_to_previous',
                       'store_latitude', 'store_longitude', '合計全天日射量(MJ/㎡)', '降雪量合計(cm)',
                       '降水量の合計(mm)', '日照時間(時間)']
    df.drop(columns_to_drop, axis=1, inplace=True)
    df.fillna(0, inplace=True)
    categorical_features = ['prefectures', 'municipality', 'shop_name_1', 'super', '天気', 'day_of_week']
    df = pd.get_dummies(df, columns=categorical_features)

    df_shop_list = pd.read_csv('data/input/shop_list.csv', encoding='utf-8')
    
    
    epochs = 100
    num_shop = len(df_shop_list)

    for epoch in range(epochs):
        #for i in range(20,num_shop):
        for i in range(20,30):
            super_name = df_shop_list['super'][i]
            shop_name = df_shop_list['shop_name_1'][i]
            df_temp = df.loc[df["super_" + super_name] & df["shop_name_1_" +shop_name]]
            print(f"Training model for shop: {super_name}, {shop_name}")

            # filling_rate列をターゲットとして選択
            target_column = 'filling_rate'
            target_index = df_temp.columns.get_loc(target_column)
            target_column2 = 'amount_kg'
            target_index2 = df_temp.columns.get_loc(target_column2)



            # # PyTorchのテンソルに変換        
            scaler = MinMaxScaler(feature_range=(-1, 1))
            df_temp = scaler.fit_transform(df_temp)
            tensor_input = torch.FloatTensor(df_temp)
            tensor_output = torch.FloatTensor(df_temp[:, target_index])

            train_inout_seq = create_inout_sequences(tensor_input, tensor_output, (target_index, target_index2))


            # モデルのインスタンス化
            model = LSTM(input_size=df.columns.size)

            # 損失関数とオプティマイザ
            loss_function = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

            for seq, labels in train_inout_seq:
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))

                y_pred = model(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

        print(f'epoch: {epoch:3} loss: {single_loss.item():10.10f}')
        # 予測値と実際のラベルを格納するリスト
        predictions = []
        actuals = []

        # モデルを評価モードに設定
        model.eval()

        # inout_seqからランダムに100個の要素を選び出す
        train_inout_seq2 = random.sample(train_inout_seq, 50)

        # 予測値と実際のラベルを取得
        for seq2, labels2 in train_inout_seq2:
            with torch.no_grad():
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
                y_pred = np.array(model(seq2))
                predictions.append(y_pred)
                actuals.append(labels2)
                #print(np.array(y_pred).shape,  np.array(labels2).shape)
                #print(np.array(seq)[column_position].shape,  np.array(labels)[column_position].shape)

        # R2スコアとRMSEを計算
        print(np.array(predictions).shape, np.array(actuals).shape)
        r2 = r2_score(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))

        print(f'R2 score: {r2}')
        print(f'RMSE: {rmse}')