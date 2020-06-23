from sklearn.tree import DecisionTreeClassifier
from env import make4
import util

x_train, x_test, y_train, y_test = make4.get_datas()

def tree_predict(depth, x, y):
    tree = DecisionTreeClassifier(max_depth=depth, criterion='gini', random_state=41)
    tree.fit(x, y)
    return tree

f = tree_predict(3, x_train, y_train)
util.show_score(f, x_test, y_test, 'Test')