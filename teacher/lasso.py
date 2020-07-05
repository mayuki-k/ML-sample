from sklearn import linear_model
from env import make14
import util

X_train, X_test, y_train, y_test = make14.get_datas()

def get_lasso(x, y):
    lasso = linear_model.Lasso(alpha=0.01, max_iter=2000, random_state=0)
    lasso.fit(x, y)
    return lasso

f = get_lasso(X_train, y_train)
util.show_score(f, X_train, y_train, 'Train')
util.show_score(f, X_test, y_test, 'Test')