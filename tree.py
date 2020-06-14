from sklearn.tree import DecisionTreeClassifier
from env import make4

x_train, x_test, y_train, y_test = make4.get_datas()

def tree_predict(depth, x, y):
    tree = DecisionTreeClassifier(max_depth=depth, criterion='gini', random_state=41)
    tree.fit(x, y)
    return tree

def show_score(tree, x, y):
    print(f'accuracy: {tree.score(x, y):.3f}')

#f = tree_predict(3, x_train, y_train)

