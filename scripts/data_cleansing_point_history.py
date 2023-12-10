from itertools import islice
import pandas as pd
import numpy as np


def process_chunk(df):
    """
    arges:
        df: DataFrame
    return:
        df: DataFrame
    【列削除】
    shop_url、free_text、rps_target_store、collect_item（必要ない）
    record_id（Nの行多い）
    other_shop_id, deactivated_flg, is_search_result_display（不明なデータのため）

    【行削除】
    item_idが1以外の行（古紙以外はcoinの付与がない、今回はぐるっとポンに絞る）
    superまたはuser_idにNanあり
    """
    # 列削除
    df = df.drop(columns=['shop_url', 'free_text', 'rps_target_store', 'collect_item', 'record_id',
                          'other_shop_id', 'deactivated_flg', 'is_search_result_display'])
    print(df.columns)

    # 行削除
    df[(df['item_id'] == 1) | (df['item_id'] == 'N')]  # 古紙データとポイント利用データを抽出
    df = df.dropna(subset=['super', 'user_id'])
    df = df[df['user_id'] != 'N']    # ユーザidがNの行を削除

    return df


def create_cleansing_csv():
    chunk_size = 10e5  # 一度に読み込む行数
    chunk_df = pd.DataFrame()  # 各チャンクを保存するためのdataframe

    count = 0
    for chunk in pd.read_csv('data/input/point_history.csv', chunksize=chunk_size):
        # ここで各チャンクに対してデータクレンジング処理を行う
        # 例：欠損値の処理、型変換、フィルタリングなど
        df = process_chunk(chunk)

        # 処理済みのチャンクをリストに追加
        chunk_df = pd.concat([chunk_df, df], ignore_index=True)

    # 最終的なDataFrameをCSVファイルとして保存
    chunk_df.to_csv('data/input/point_history_cleansing.csv', index=False)


def isunique_columns(df, column1, column2):
    # column1とcolumn2の組み合わせで新しいカラムを作成
    df['combined'] = df[column1].astype(str) + '_' + df[column2].astype(str)

    # 新しいカラムのユニークな値を取得
    unique_combinations = df['combined'].unique()

    print(unique_combinations)
    df = df.drop(columns=['combined'])


if __name__ == '__main__':
    # 表示する最大列数を設定（例: Noneは無制限を意味します）
    pd.set_option('display.max_columns', 100)
    # 表示する最大行数も設定できます（オプション）
    pd.set_option('display.max_rows', 100)

    # create_cleansing_csv()

    df = pd.read_csv('data/input/point_history_cleansing.csv')
    # print(df['status'].unique())
    # tmpDf = df[df['item_id'] == 'N']
    # print(tmpDf.head())  # TODO: user_idが69はテストユーザーっぽい。他にもあるか探して除外したい
    # tmpDf = df[df['status'] == 3]
    # print(tmpDf)

    df = df.replace('N', np.nan)

    df['use_date'] = pd.to_datetime(df['use_date'], errors='coerce')
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
    df['created_at_1'] = pd.to_datetime(df['created_at_1'], errors='coerce')
    df['updated_at_1'] = pd.to_datetime(df['updated_at_1'], errors='coerce')
    # time型に変換
    df['store_opening_time'] = pd.to_datetime(
        df['store_opening_time'], format='%H:%M:%S').dt.time
    df['store_closing_time'] = pd.to_datetime(
        df['store_closing_time'], format='%H:%M:%S').dt.time
    df['rps_opening_time'] = pd.to_datetime(
        df['rps_opening_time'], format='%H:%M:%S').dt.time
    df['rps_closing_time'] = pd.to_datetime(
        df['rps_closing_time'], format='%H:%M:%S').dt.time

    column_types = {
        'user_id': int,
        'amount': np.float16,
        'amount_kg': np.float16,
        'point': np.float16,
        'total_point': np.float16,
        'total_amount': np.float16,
        'coin': np.float16,
        'id_1': 'Int64',
        'series': 'Int64',
        # 'rank_id': 'Int64',

    }
    df = df.astype(column_types)

    # print(df[df['series_id'].isna() & ~(df['series'].isna())])
    # print(df[df['use_date'] > pd.to_datetime('2023-12-06')])
    # print(df[df['created_at'] > pd.to_datetime('2023-12-06')])
    # print(df[df['updated_at'] > pd.to_datetime('2023-12-06')])
    # print(df[df['created_at_1'] > pd.to_datetime('2023-12-06')])
    # print(df[df['updated_at_1'] > pd.to_datetime('2023-12-06')])
    # print(df['rank_id'].unique())
    # print(df[df['rank_id'].isna()])
    # TODO: rank_idが「nan」のものは削除？QAの返答待ち
    # print(df[df['status'] == 7]['rank_id'].unique())

    # 列名を直感的に変更
    df = df.rename(columns={'id_1': '支店ID'})
    df = df.rename(columns={'item_id': 'リサイクル分類ID'})

    # 不正な行削除
    df = df[(df['amount'] >= 0) | df['amount'].isna()]    # amount(持ち込み量)が負の値を削除
    df = df[(df['amount_kg'] >= 0) | df['amount_kg'].isna()]    # amount_kg(持ち込み量kg)が負の値を削除
    df = df[(df['point'] >= 0) | df['point'].isna()]    # point(RPSのポイント)が負の値を削除 TODO: QAの返答待ち
    df = df[(df['total_point'] >= 0) | df['total_point'].isna()]    # total_point(RPSのポイント)が負の値を削除
    # 列削除
    df = df.drop(columns=['unit_id', 'prefectures', 'municipality', 'series','address'])



    
    # csv書き出し
    df.to_csv('data/input/point_history_cleansing_2.csv', index=True)

