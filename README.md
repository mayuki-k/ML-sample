# ML-sample
機械学習

|ファイル名|説明|
| --- | --- |
|logistic.py|ロジスティック回帰|
|svc.py|SVC|

# 三点セット

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

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

# matplotlib(plt)

## グラフ表示

```
plt.show()
```

```
plt.scatter(X1, X2, c=Y, cmap=plt.cm.coolwarm)
```

# mlxtend.plotting

## plot_decision_regions

```
plot_decision_regions(X, Y, clf, res)
```

# Question

- 正規化、標準化とは？