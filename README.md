# ML-sample
機械学習

|ファイル名|説明|
| --- | --- |
|logistic.py|ロジスティック回帰|
|svc.py|SVC|

# docker-jupyternotebook 起動

```
docker run -d --name note -p 8888:8888 jupyter/datascience-notebook
```

その後、Tokenによってpasswordのセットアップが必要なので注意

コンテナ内に入り

```
jupyter notebook list
```

で、tokenの確認可能