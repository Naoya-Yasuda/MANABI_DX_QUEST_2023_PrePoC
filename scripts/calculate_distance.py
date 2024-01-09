import pandas as pd
import requests
import os
import time
from tqdm import tqdm
import numpy as np
# カレントディレクトリを.pyと合わせるために以下を実行
from pathlib import Path
if Path.cwd().name == "notebook":
    os.chdir("..")


def haversine(lat1, lon1, lat2, lon2):
    # print(lat1, lon1, lat2, lon2)
    # 地球の半径（キロメートル）
    R = 6371.0

    # 緯度と経度をラジアンに変換
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # ハーヴァーサイン公式
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


# CSVファイルを読み込む
df = pd.read_csv('data/input/user_info_longitude_latitude.csv')
df2 = pd.read_csv('data/input/ユーザー基本情報_2023-12-21.csv', encoding='sjis')
df = pd.merge(df, df2, left_on='id', right_on='利用者ID', how='left')
shop_df = pd.read_csv('data/input/shop_list.csv')
# dfのスーパーカラムと登録店舗カラムそれぞれとshop_dfのsuper,shop_name_1カラムを比較して、shop_id、store_latitude,store_longitudeを取得する
merged_df = pd.merge(df, shop_df, left_on=['スーパー', '登録店舗'], right_on=[
                     'super', 'shop_name_1'], how='inner')

# マージ結果を確認
print(merged_df)
# dfの経度と緯度とshop_dfの経度と緯度を比較して、距離を計算する
# ハーヴァーサイン公式を使用して距離を計算する関数

for index, row in tqdm(merged_df.iterrows()):
    df.at[index, '登録店舗との距離'] = haversine(
        row['緯度'], row['経度'], row['store_latitude'], row['store_longitude'])

print(df)
df.to_csv('user_info_longitude_latitude.csv',
          mode='w', header=True, index=False)
