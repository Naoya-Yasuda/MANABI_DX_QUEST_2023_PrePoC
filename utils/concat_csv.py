import pandas as pd
import os
import glob
import sys

# 親ディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 指定フォルダ内のすべてのcsvファイルを検索
folder_path = 'data/input/回収予定表/*/'
files = glob.glob(os.path.join(folder_path, '*.csv'))

df = pd.DataFrame()
for file in files:
    df = pd.concat([df, pd.read_csv(file, encoding='utf-8')], ignore_index=True)

# CSVファイルに保存
df.to_csv('data/references/RS_twice_collection_day.csv', index=False, encoding='utf-8')