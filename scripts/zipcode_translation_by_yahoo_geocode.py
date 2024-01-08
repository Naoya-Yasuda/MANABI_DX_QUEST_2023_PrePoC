import pandas as pd
import requests
from tqdm import tqdm
import json
import os

# CSVファイルを読み込む
df = pd.read_csv('data/input/user_info_longitude_latitude.csv')

# Yahoo!ジオコーダAPIを使用して経度と緯度を取得する関数


def get_coordinates(zipcode, api_key):
    url = f'https://map.yahooapis.jp/search/zip/V1/zipCodeSearch?appid={api_key}&query={zipcode}&output=json'
    response = requests.get(url)
    # レスポンスのステータスコードと内容を確認
    if response.status_code != 200:
        raise Exception(
            f"API request failed with status code {response.status_code}, Response: {response.text}")

    data = response.json()
    # JSONデコード時のエラーをキャッチ
    try:
        data = response.json()
    except json.JSONDecodeError:
        raise Exception(
            f"Error: Unable to decode JSON response, Response: {response.text}")

    # Yahoo!ジオコーダAPIのレスポンス形式に基づいて経度と緯度を取得
    # 以下のコードはレスポンスの形式によって異なる可能性がある
    if data['ResultInfo']['Count'] > 0:
        print('zipcode: ', zipcode)
        # print(data)
        coordinates = data['Feature'][0]['Geometry']['Coordinates'].split(
            ',')
        return coordinates[1], coordinates[0]  # 緯度, 経度
    else:
        # 結果が0の場合
        return None, None


# APIキーを設定（実際のAPIキーに置き換えてください）
api_key = 'YOUR_API_KEY_HERE'  # 'YOUR_API_KEY_HERE'の部分を置き換える

# 処理済みのzipcodeを記録するファイル
processed_zipcodes_file = 'processed_zipcodes.txt'

# 処理済みのzipcodeを読み込む
if os.path.exists(processed_zipcodes_file):
    with open(processed_zipcodes_file, 'r') as file:
        processed_zipcodes = set(file.read().splitlines())
else:
    processed_zipcodes = set()


# 条件に合う行を絞り込む
'''郵便番号（zipcode）が 'N' でなく、かつ zipcode が空（NaN）でない行を選択。
経度（経度）と緯度（緯度）の両方が空（NaN）ではない行を除外。'''
cond = ((df['zipcode'] != 'N') & ~pd.isna(df['zipcode'])) & (
    pd.isna(df['経度']) | pd.isna(df['緯度']))
subset_df = df[cond]

counter = 0
# APIから緯度と経度を取得し、結果をキャッシュする辞書
coordinates_cache = {}
# 処理中のzipcodeを追跡するための一時リスト
current_processed_zipcodes = []

# 絞り込んだデータフレームに対して処理
for index, row in tqdm(subset_df.iterrows()):
    zipcode = row['zipcode']

    # # すでに処理されたzipcodeはスキップ
    # if zipcode in processed_zipcodes:
    #     continue

    # キャッシュにすでに存在する場合は、その値を使用
    if zipcode in coordinates_cache:
        lat, lon = coordinates_cache[zipcode]
    else:
        lat, lon = get_coordinates(zipcode, api_key)
        coordinates_cache[zipcode] = (lat, lon)
    if lat and lon:
        df.at[index, '緯度'] = float(lat)
        df.at[index, '経度'] = float(lon)
        counter += 1
        # 処理済みリストに追加
        processed_zipcodes.add(zipcode)
        current_processed_zipcodes.append(zipcode)

        # 一定回数ごとにファイルに追記
        if counter % 100 == 0:
            df.to_csv('updated_file.csv', mode='w', header=False, index=False)
            with open(processed_zipcodes_file, 'a') as file:
                file.write('\n'.join(current_processed_zipcodes) + '\n')
            current_processed_zipcodes = []  # 一時リストをリセット

# 最後に残りの部分をファイルに追記
df.to_csv('updated_file.csv', mode='w', header=False, index=False)
with open(processed_zipcodes_file, 'w') as file:
    file.write('\n'.join(processed_zipcodes) + '\n')
