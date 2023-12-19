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
4. `git rm --cached -r .` `git reset --hard` キャッシュ削除


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
dataフォルダは、input/outputに分けています。

## input
inputフォルダには、元データのcsvファイルや、クレンジング後のcsvファイルを配置しています。

## output
outputフォルダには、提出用のcsvファイルを出力します。ファイル名は「sub_(year-month-day-hour-min)_(score)」のように設定し、後述するログと照合できるようにしています。

## features
featuresフォルダには、train/testから作成した各特徴量を保存しています。

## logs
logsフォルダには、計算の実行ごとに下記の情報などを出力しています。ファイル名は「log_(year-month-day-hour-min).log」のように設定し、前述した通り提出用のcsvファイルと照合できるようにしています。

## notebook
notebookフォルダには、探索的データ分析などで利用したJupyter Notebookを配置しています。ここで試行錯誤した結果を、適切なフォルダ内のpythonファイルに取り込んでいきます。

## scripts
scriptsフォルダには、汎用的なpythonファイルを配置します。例えば convert_to_feather.py ファイルは、csvファイルをfeather形式のファイルに変換します。

## utils
utilsフォルダには、汎用的に使える関数を書いています。

## 利用した特徴量
- trainのshape
- 学習器のパラメータ
- cvのスコア

## 参考
[upuraのブログ](https://upura.hatenablog.com/entry/2018/12/28/225234)



