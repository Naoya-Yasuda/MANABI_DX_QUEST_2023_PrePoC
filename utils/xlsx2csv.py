import pandas as pd
import os
import glob
import sys

# 親ディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 指定フォルダ内のすべてのExcelファイルを検索
folder_path = 'data/input/回収予定表/*/'
excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))

for excel_file in excel_files:
    # 出力CSVファイルのパス
    output_csv = os.path.splitext(excel_file)[0] + '.csv'

    # Excelファイルを読み込む
    xls = pd.ExcelFile(excel_file)

    # CSVファイルに書き出すための空のDataFrameを作成
    combined_csv = pd.DataFrame()

    # すべてのシートをループ処理
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        df['年月日'] = pd.to_datetime(df.iloc[:, 0]).dt.strftime('%Y/%m/%d')
        df['super'] = sheet_name

        # 'Unnamed: 0'列が存在する場合は削除
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        df = df.melt(id_vars=['年月日', 'super'], var_name='shop_name_1', value_name='flag')
        combined_csv = pd.concat([combined_csv, df], ignore_index=True)

    combined_csv.fillna(0, inplace=True)
    combined_csv['flag'] = combined_csv['flag'].astype(int)

    # CSVファイルに保存
    combined_csv.to_csv(output_csv, index=False, encoding='utf-8-sig')