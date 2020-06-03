# このモジュールでは学習のみを行う。データ取得は別モジュールにて。

from sklearn import linear_model
from sklearn.model_selection import train_test_split
# データ取得モジュール
from env import make1

label, data = make1.get_data()

def logistic():
    x_train, x_test, y_train, y_test = train_test_split(data, label, random_state=0)
    pre = linear_model.LogisticRegression(random_state=0)
    pre.fit(x_train, y_train)
    print(f'score = {pre.score(x_test, y_test)}')
    print(pre.predict([[22, 8]]))

logistic()