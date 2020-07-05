from sklearn.ensemble import RandomForestClassifier
from env import make4

x_train, x_test, y_train, y_test = make4.get_datas()

def r_forest(x, y, est, features, depth, rand):
    forest = RandomForestClassifier(n_estimators=est, 
                                max_features=features, 
                                max_depth=depth,
                                criterion='gini', 
                                random_state=rand)
    forest.fit(x_train, y_train)
    return forest

def show_score(f, x, y):
    print(f'Accuracy: {f.score(x, y):.3f}')

f = r_forest(x_train, y_train, 7, 3, 3, 41)
show_score(f, x_train, y_train)
