# PatchWorks_PrePoC
【マナビDXクエスト】PatchWorksのプレPoC用のレポジトリです。

# 環境構築
conda create -n patchworks python=3.11.5  
conda activate patchworks  
pip install -r requirements.txt  

# フォルダ構成
## configs
jsonファイルで、諸設定を記載しています。  
記載している情報は「利用している特徴量」「学習器のパラメータ」などです。  

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
inputフォルダには、元データのcsvファイルや、feather形式に変換したファイルなどを配置しています。

## output
outputフォルダには、提出用のcsvファイルを出力します。ファイル名は「sub_(year-month-day-hour-min)_(score)」のように設定し、後述するログと照合できるようにしています。

## features
featuresフォルダには、train/testから作成した各特徴量を保存しています。

## logs
logsフォルダには、計算の実行ごとに下記の情報などを出力しています。ファイル名は「log_(year-month-day-hour-min).log」のように設定し、前述した通り提出用のcsvファイルと照合できるようにしています。

## 利用した特徴量
- trainのshape
- 学習器のパラメータ
- cvのスコア

## 参考
[upuraのブログ](https://upura.hatenablog.com/entry/2018/12/28/225234)



