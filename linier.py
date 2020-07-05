from sklearn import linear_model
from env import make14
import util

X_train, X_test, y_train, y_test = make14.get_datas()

X_train_single = X_train[:, 5].reshape(-1, 1)
X_test_single = X_test[:, 5].reshape(-1, 1)

def get_linear_single(x, y):
    linear = linear_model.LinearRegression()
    linear.fit(x, y)
    return linear

def get_linear(x, y):
    linear = linear_model.LinearRegression()
    linear.fit(x, y)
    return linear

linear = get_linear(X_train, y_train)
util.show_score(linear, X_train, y_train, 'Train')
util.show_score(linear, X_test, y_test, 'Test')
