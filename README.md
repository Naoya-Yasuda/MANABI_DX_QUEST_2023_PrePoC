# PatchWorks_PrePoC
【マナビDXクエスト】PatchWorksのプレPoC用のレポジトリです。

# 環境構築
conda create -n patchworks python=3.11.5
conda activate patchworks
pip install -r requirements.txt
## notebookの実行結果をGit管理から外す
notebookの実行結果は実行する度に差分として出てしまうので差分が出ないように管理から外します。
1. `conda activate patchworks` 実施していなければする
2. `pip install nbstripout` nbstripoutをインストール
3. `nbstripout --install` nbstripout の Git への設定
4. `git rm --cached -r .` キャッシュ削除
5. `git reset --hard` キャッシュ削除


# フォルダ構成
## configs
jsonファイルで、諸設定を記載しています。
記載している情報は「利用している特徴量」「学習器のパラメータ」などです。

例：
{
  "features": [
      "age",
      "embarked",
      "family_size",
      "fare",
      "pclass",
      "sex"
  ],
  "lgbm_params": {
    "learning_rate": 0.1,
    "num_leaves": 8,
    "boosting_type": "gbdt",
    "colsample_bytree": 0.65,
    "reg_alpha": 1,
    "reg_lambda": 1,
    "objective": "multiclass",
    "num_class": 2
  },
  "loss": "multi_logloss",
  "target_name": "Survived",
  "ID_name": "PassengerId"
}

またコンペのデータに依存するカラム名なども、このjsonファイルから読み取る形式にしています。

## data
dataフォルダは、input/output/references/appendicesに分けています。

### input
inputフォルダには、元データのcsvファイルや、クレンジング後のcsvファイルを配置しています。

### output
outputフォルダには、モデルの予測結果をcsvファイルとして出力します。

### references
referencesフォルダには、2回回収日やリサイクルステーションが少ない店など、参照用データを保存します。

### appendices
データ定義書などの付録データ、その他受領したxlsxやpdfファイルを保存します。

## logs
logsフォルダには、計算の実行ごとに下記の情報などを出力しています。ファイル名は「log_(year-month-day-hour-min).log」のように設定し、前述した通り提出用のcsvファイルと照合できるようにしています。

## notebook
notebookフォルダには、探索的データ分析などで利用したJupyter Notebookを配置しています。ここで試行錯誤した結果を、適切なフォルダ内のpythonファイルに取り込んでいきます。

## scripts
scriptsフォルダには、汎用的なpythonファイルを配置します。例えば convert_to_feather.py ファイルは、csvファイルをfeather形式のファイルに変換します。

## utils
utilsフォルダには、汎用的に使える関数を書いています。

## RS_filling_rate
RS_filling_rateフォルダでは、充填率予測に向けたコードを配置しています。 
以下の順で実行してください。
1. weather_data_fetcher.py
2. RS_fillingrate_test.py
3. RS_filling_time_detecter.py

visualizationと先頭に付くファイルは、データ可視化用のファイルです。
実行の有無は他のファイルに影響を与えません。


## 参考
[upuraのブログ](https://upura.hatenablog.com/entry/2018/12/28/225234)



