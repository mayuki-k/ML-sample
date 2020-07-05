# このモジュールでは学習のみを行う。データ取得は別モジュールにて。

from sklearn import linear_model
# データ取得モジュール
from env import make1
import util

x_train, x_test, y_train, y_test = make1.get_data()

def logistic(x, y):
    pre = linear_model.LogisticRegression(random_state=0)
    pre.fit(x, y)
    return pre

f = logistic(x_train, y_train)
util.show_score(f, x_test, y_test, 'Test')