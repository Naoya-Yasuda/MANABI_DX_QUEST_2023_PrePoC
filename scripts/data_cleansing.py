"""
 このファイルは、前処理としてデータクレンジングを行うためのファイルです。
　TODO：user_info.csvの不正データ洗い出し
 TODO:club_coinカラムに-の値が18個あり。QA返答待ち
 TODO:recycle_pointカラムに-の値が9個あり。QA返答待ち
 TODO:
"""
from itertools import islice
import pandas as pd
import numpy as np


def check_integer(df, column_name_list):
    """
    float型やstr型で整数でない値があるかどうかを確認する。
    """
    for column_name in column_name_list:
        unique_values = df[column_name].unique()
        is_integer = True
        for x in unique_values:
            if (isinstance(x, float) and not x.is_integer()) or (isinstance(x, str) and not x.isdigit()):
                print(f"整数でない{column_name}: {x}")
                is_integer = False

        # is_integer = all(x.is_integer() for x in unique_values)
        print(f'{column_name} is int?: ', is_integer)


def fix_csv_file(input_file, output_file, chunk_size=30000):
    # point_history.csvの行が壊れているため、読み込めない。csvファイルを修正する必要がある。
    # ファイル内の不正な行を修正する。行の「"N,」を「"N",」に変更する。
    """
    input_file: str 修正するファイルのパス
    output_file: str 修正したファイルのパス
    chunk_size: int 一度に読み込む行数
    """
    total_lines_fixed = 0  # 修正した合計行数を追跡

    with open(output_file, 'w', encoding='utf-8') as outfile:
        with open(input_file, 'r', encoding='utf-8') as infile:
            while True:
                lines = list(islice(infile, chunk_size))
                if not lines:
                    break

                fixed_lines = []
                for line in lines:
                    # "N, の場合を "N", に置換
                    line = line.replace('"N,', '"N",')
                    # 行末尾が '"N' で終わる場合、それを '"N"' に置換
                    if line.endswith('"N\n'):
                        line = line[:-2] + '"\n'
                    fixed_lines.append(line)

                outfile.writelines(fixed_lines)

                total_lines_fixed += len(fixed_lines)
                print(f'{total_lines_fixed}行のデータを修正しました。')

# 以下3行をファイル修正時にコメントを外して実行する
# input_file = 'data/input/point_history_old.csv'  # 入力ファイルのパス
# output_file = 'data/input/fixed_point_history.csv'  # 出力ファイルのパス
# fix_csv_file(input_file, output_file)


# assets配下のcsvデータをそれぞれ読み込む
userDf = pd.read_csv('data/input/user_info.csv')
# Nが入っているため、nanに変換してから型を統一する→nanで処理できるモデルがあるため
userDf = userDf.replace('N', np.nan)

# check_integer(userDf, ['club_coin', 'recycle_point',
#                        'recycle_amount_after_gold_member', 'zipcode'])

userDf['birth_day'] = pd.to_datetime(userDf['birth_day'], errors='coerce')
column_types = {
    'total_recycle_amount': float,
    'recycle_amount_per_year': float,
    'rank_continuation_class': int,
    'zipcode': str
}
# zipcodeがfloatgata型になっているため、「.0」が付与されている。これを削除する。
userDf['zipcode'] = userDf['zipcode'].apply(
    lambda x: str(float(x)).replace('.0', ''))
# 長さが7桁でないzipcodeをNaNに置換する処理
userDf['zipcode'] = userDf['zipcode'].apply(
    lambda x: np.nan if len(x) != 7 else x)
print('----------1-----------')

# 'birth_day'が'1910-01-01'以前の行をフィルタリング 31行
filtered_by_birth_day = userDf[userDf['birth_day'] <= '1910-01-01']
# TODO: birth_dayをどうするか検討中
# birth_day_rows = userDf['birth_day'].apply(
#     lambda x: np.NaN if pd.to_datetime(x) < pd.to_datetime('1910-01-01') else x)

# userDf['birth_day'] = pd.to_datetime(userDf['birth_day']).dt.date

# print(filtered_by_birth_day)

# inaffected_coinは全てNaNを確認済み
# TODO: カラム削除する
filtered_by_inaffected_coin = userDf[userDf['inaffected_coin'].notna()]
print(filtered_by_inaffected_coin)

# 性別は三種類(男・女・無回答)を確認ずみ
# TODO: 区分値に変更予定　draftとして0, 1, 2の区分値を持つ単一のカラムに変更する。
# 男カラム・女カラムの2つ（ワンホットエンコーディング）は複雑な機械学習モデルを使用する場合に必要になるかも
userDf['gender'] = userDf['gender'].apply(
    lambda x: 0 if x == '男' else (1 if x == '女' else 2))
print(userDf['gender'].unique())

# data2 = pd.read_csv('data/input/point_history.csv')
# print(data2)
# print(data2.shape)

# data3 = pd.read_csv('data/input/gacha_history.csv')
# print(data3)
# print(data3.shape)
