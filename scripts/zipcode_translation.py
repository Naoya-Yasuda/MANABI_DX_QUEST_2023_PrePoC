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


def batch_geocode(zipcode):
    """APIを使用して郵便番号のリストをジオコードする"""
    response = requests.get(url + zipcode)
    data = response.json()
    if response.status_code != 200:
        print(f"エラー: {response.status_code, data, params}")
    # 'response' と 'location' キーの存在を確認
    if 'response' not in data or 'location' not in data['response']:
        return np.nan, np.nan
    # 空の時の処理
    if not data['response'] or not data['response']['location']:
        return np.nan, np.nan

#     print(data['response']['location'][0]['x'], data['response']['location'][0]['y'])
    return data['response']['location'][0]['x'], data['response']['location'][0]['y']


if __name__ == '__main__':
    # CSVファイルからDataFrameを読み込む
    df = pd.read_csv('data/input/user_info_cleansing.csv',
                     dtype={'zipcode': str})

    # 欠損値を「N」に置換
    df['zipcode'] = df['zipcode'].fillna('N')

    url = 'http://geoapi.heartrails.com/api/json?method=searchByPostal&postal='

    # APIリクエストをバッチで実行
    for i in tqdm(range(0, len(df))):
        # 'N'を含まない郵便番号を作成
        zipcode = [zipcode for zipcode in df['zipcode']
                   [i:i + 1] if zipcode != 'N']
    #     display(zipcode)
        if zipcode:
            zipcode = zipcode[0]
            # 緯度と経度のデータを取得
            latitude, longitude = batch_geocode(zipcode)

            # 新しいカラムをDataFrameに追加して値を設定
            df.loc[df['zipcode'] == zipcode, '経度'] = float(longitude)
            df.loc[df['zipcode'] == zipcode, '緯度'] = float(latitude)

        if i % 5 == 0:
            time.sleep(1)
    display(df)
    df.to_csv('data/input/user_info_longitude_latitude.csv', index=True)
