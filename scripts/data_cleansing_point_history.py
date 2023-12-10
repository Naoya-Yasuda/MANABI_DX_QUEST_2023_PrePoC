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
    df = df[df['item_id'] == 1]    # 古紙データに絞り込み
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
        #chunks.append(processed_chunk)
        #chunks.append(chunk)
        #df.to_csv('data/input/point_history_cleansing_'+str(count)+'.csv', index=False)    
        #count += 1
        chunk_df = pd.concat([chunk_df, df], ignore_index=True)

    # 最終的なDataFrameをCSVファイルとして保存
    chunk_df.to_csv('data/input/point_history_cleansing.csv', index=False)


if __name__ == '__main__':
    df =pd.read_csv('data/input/point_history_cleansing.csv')
    df = df.replace('N', np.nan)

    df['use_date'] = pd.to_datetime(df['use_date'], errors='coerce')
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
    df['created_at_1'] = pd.to_datetime(df['created_at_1'], errors='coerce')
    df['updated_at_1'] = pd.to_datetime(df['updated_at_1'], errors='coerce')

    column_types = {
            'user_id': int,  
            'amount': np.float16,
            'amount_kg': np.float16,
            'point': np.float16,
            'total_point': np.float16,
            'total_amount': np.float16,
            'coin': np.float16,
            # 'id_1': int,
            # 'series': int,


        }
    df = df.astype(column_types)
    # print(df.info())
    print(df[df['total_amount'] < 0])

    # 不正な行削除
    # df = df[df['amount'] >= 0]    # amount(持ち込み量)が負の値を削除
