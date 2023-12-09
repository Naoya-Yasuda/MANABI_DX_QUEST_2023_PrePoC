"""
 このファイルは、前処理としてデータクレンジングを行うためのファイルです。
 TODO：zipcodeの8桁以上のデータ、「.」が入っているデータをnanにする
　TODO：user_info.csvの不正データ洗い出し
"""
from itertools import islice
import pandas as pd


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
userDf = userDf.replace('N', float('nan'))
print('------------------- 1 ----------------------')
print(userDf.info())
check_integer(userDf, ['club_coin', 'recycle_point',
                       'recycle_amount_after_gold_member', 'zipcode'])

userDf['birth_day'] = pd.to_datetime(userDf['birth_day'], errors='coerce')
column_types = {
    'total_recycle_amount': float,
    'recycle_amount_per_year': float,
    'rank_continuation_class': int,
}

userDf = userDf.astype(column_types)
print('------------------- 2 ----------------------')
print(userDf.info())

# print(userDf)
# print(userDf.shape)


# data2 = pd.read_csv('data/input/point_history.csv')
# print(data2)
# print(data2.shape)

# data3 = pd.read_csv('data/input/gacha_history.csv')
# print(data3)
# print(data3.shape)
