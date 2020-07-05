from sklearn import linear_model
from env import make14
import util

X_train, X_test, y_train, y_test = make14.get_datas()

def get_ridge(x, y):
    ridge = linear_model.Ridge(alpha=1.0, random_state=0)
    ridge.fit(x, y)
    return ridge

f = get_ridge(X_train, y_train)
util.show_score(f, X_train, y_train, 'Train')
util.show_score(f, X_test, y_test, 'Test')