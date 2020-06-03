from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import svm

# データの取得
from env import make2

x, y = make2.get_datas()

def standalization(data):
    sc = preprocessing.StandardScaler()
    sc.fit(data)
    return sc.transform(data)

def support_vector_classification(x, y):
    f = svm.LinearSVC(random_state=0)
    f.fit(x, y)
    print(f.predict([[0, 1]]))

x = standalization(x)
support_vector_classification(x, y)



