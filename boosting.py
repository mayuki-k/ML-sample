from sklearn import ensemble
from env import make9
import util

x_train, x_test, y_train, y_test = make9.get_datas()



def get_gb(x, y):
    gb = ensemble.GradientBoostingClassifier(n_estimators=500, random_state=0)
    gb.fit(x, y)
    return gb

def get_gbes(x, y):
    gbes = ensemble.GradientBoostingClassifier(n_estimators=500, validation_fraction=0.25, n_iter_no_change=5, random_state=0)
    gbes.fit(x, y)
    return gbes

gb = get_gbes(x_train, y_train)
util.show_score(gb, x_test, y_test, 'Test')