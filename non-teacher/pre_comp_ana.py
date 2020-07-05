# 主成分分析

from sklearn import preprocessing, decomposition
from env import make6

x = make6.get_data()



def get_standarization(x):
    sc = preprocessing.StandardScaler()
    sc.fit(x)
    return sc

def get_pca(x_std):
    pca = decomposition.PCA(random_state=0)
    pca.fit(x)
    return pca

def do_pca(sc, pca, data):
    data_std = sc.transform([data])
    data_pca = pca.transform(data_std)
    print(data_pca)

def show_pca_result(pca):
    print('-----主成分の分散説明率-----')
    print(pca.explained_variance_ratio_)

    print('-----固有ベクトル-----')
    print(pca.components_)

sc = get_standarization(x)
x_std = sc.transform(x)
pca = get_pca(x_std)
x_pca = pca.transform(x_std)

show_pca_result(pca)
