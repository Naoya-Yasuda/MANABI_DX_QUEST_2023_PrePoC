from itertools import islice
import pandas as pd
import numpy as np
import dask.dataframe as dd

import dask.dataframe as dd
import numpy as np
import pandas as pd

def process_chunk_dask(df):
    # 列削除
    df = df.drop(['shop_url', 'free_text', 'rps_target_store', 'collect_item', 'record_id',
                  'other_shop_id', 'deactivated_flg', 'is_search_result_display', 'unit_id', 'series', 'address'], axis=1)
    # NaN に置き換える
    df = df.replace('N', np.nan)
    df = df.replace('NaN', np.nan)
    df = df.replace('nan', np.nan)
    df = df.replace('foo', np.nan)

    # データ型を変更
    df['item_id'] = df['item_id'].apply(pd.to_numeric, errors='coerce')
    df['amount'] = df['amount'].apply(pd.to_numeric, errors='coerce')
    df['amount_kg'] = df['amount_kg'].apply(pd.to_numeric, errors='coerce')
    df['point'] = df['point'].apply(pd.to_numeric, errors='coerce')
    df['total_point'] = df['total_point'].apply(pd.to_numeric, errors='coerce')
    #df = df.astype(dtype_changes)

    # 行削除
    df = df[df['item_id'] == 1] 
    df = df.dropna(subset=['super'])  
    df = df.dropna(subset=['amount'])  
    df = df.dropna(subset=['amount_kg'])  
    df = df.dropna(subset=['point'])  
    df = df.dropna(subset=['total_point'])  
    df = df.dropna(subset=['prefectures'])  

    # 列名を直感的に変更
    df = df.rename(columns={'id_1': '支店ID'})
    df = df.rename(columns={'item_id': 'リサイクル分類ID'})

    # 型変換
    dtype_changes = {        
       'id' : np.float32,
       'user_id' : int,
       'series_id' : np.float32,
       'shop_id' : np.float32,
       'shop_name' : str,
       'card_id' : str,
       'リサイクル分類ID' : np.float32,
       'amount' : np.float32,
       'amount_kg' : np.float32,
       'point' : np.float32,
       'total_point' : np.float32,
       'status' : np.float32,
       'total_amount' : np.float32,
       'coin' : np.float32,
       'rank_id' : np.float32,
       'use_date' :   'datetime64[ns]',
       'created_at' : 'datetime64[ns]',
       'updated_at' : 'datetime64[ns]',
       '支店ID' : np.float32,
       'super' : str,
       'prefectures' : str,
       'municipality' : str,
       'shop_name_1' :  str,
       'shop_id_1' :    str,
       'created_at_1' : 'datetime64[ns]',
       'updated_at_1' : 'datetime64[ns]',
       'store_latitude' : np.double,
       'store_longitude' : np.double,
    }



    # amount(持ち込み量)が負の値を削除
    df = df[(df['amount'] >= 0) | df['amount'].isna()]
    # amount_kg(持ち込み量kg)が負の値を削除
    df = df[(df['amount_kg'] >= 0) | df['amount_kg'].isna()]
    # point(RPSのポイント)が負の値を削除
    df = df[(df['point'] >= 0) | df['point'].isna()]
    # total_point(RPSのポイント)が負の値を削除
    df = df[(df['total_point'] >= 0) | df['total_point'].isna()]

    # Daskでは、計算を実行するためにはcompute()メソッドを呼び出す必要があります。
    # 以下のコードは、最終的なDataFrameをPandas DataFrameとして取得します。
    # pandas_df = df.compute()

    return df

def process_and_save_csv_dask(input_path, output_path):
    # Dask DataFrameの作成
    ddf = dd.read_csv(input_path, dtype='object')

    # データ加工の適用
    ddf = ddf.map_partitions(process_chunk_dask)
    #ddf = process_chunk(ddf)

    # 加工後のデータをCSVとして保存
    ddf.to_csv(output_path, index=False, single_file=True)


def process_chunk(df):
    
    # 列削除
    df = df.drop(['shop_url', 'free_text', 'rps_target_store', 'collect_item', 'record_id',
                  'other_shop_id', 'deactivated_flg', 'is_search_result_display', 'unit_id', 'series', 'address'], axis=1)
    # NaN に置き換える
    df = df.replace('N', np.nan)
    df = df.replace('NaN', np.nan)
    df = df.replace('nan', np.nan)
    df = df.replace('foo', np.nan)


    # 行削除
    df = df[df['item_id'] == 1] 
    df = df.dropna(subset=['super'])  
    df = df.dropna(subset=['amount'])  
    df = df.dropna(subset=['amount_kg'])  
    df = df.dropna(subset=['point'])  
    df = df.dropna(subset=['total_point'])  
    df = df.dropna(subset=['prefectures'])  

    # 列名を直感的に変更
    df = df.rename(columns={'id_1': '支店ID'})
    df = df.rename(columns={'item_id': 'リサイクル分類ID'})

    # 型変換
    dtype_changes = {        
       'id' : np.float32,
       'user_id' : np.float32,
       'series_id' : np.float32,
       'shop_id' : np.float32,
       'shop_name' : str,
       'card_id' : str,
       'リサイクル分類ID' : np.float32,
       'amount' : np.float32,
       'amount_kg' : np.float32,
       'point' : np.float32,
       'total_point' : np.float32,
       'status' : np.float32,
       'total_amount' : np.float32,
       'coin' : np.float32,
       'rank_id' : np.float32,
       'use_date' :   'datetime64[ns]',
       'created_at' : 'datetime64[ns]',
       'updated_at' : 'datetime64[ns]',
       '支店ID' : np.float32,
       'super' : str,
       'prefectures' : str,
       'municipality' : str,
       'shop_name_1' :  str,
       'shop_id_1' :    str,
       'created_at_1' : 'datetime64[ns]',
       'updated_at_1' : 'datetime64[ns]',
       'store_latitude' : np.double,
       'store_longitude' : np.double,
    }
    df = df.astype(dtype_changes)


    # amount(持ち込み量)が負の値を削除
    df = df[(df['amount'] >= 0) | df['amount'].isna()]
    # amount_kg(持ち込み量kg)が負の値を削除
    df = df[(df['amount_kg'] >= 0) | df['amount_kg'].isna()]
    # point(RPSのポイント)が負の値を削除
    df = df[(df['point'] >= 0) | df['point'].isna()]
    # total_point(RPSのポイント)が負の値を削除
    df = df[(df['total_point'] >= 0) | df['total_point'].isna()]

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

        chunk_df.to_csv('data/input/point_history_rps_'+ str(count) +'.csv', index=False)
        count += 1


def merge_csv():
    df = pd.DataFrame()
    for i in range(16):
        df_tmp = pd.read_csv('data/input/point_history_rps_'+ str(i) +'.csv')
        df = pd.concat([df, df_tmp], ignore_index=True)
        print(i)
    df.to_csv('data/input/point_history_rps.csv', index=False)

# 方法1 Daskを使用
#process_and_save_csv_dask('data/input/point_history.csv', 'data/input/point_history_rps.csv')

# 方法2 Pandasを使用
create_cleansing_csv()
#merge_csv() 
