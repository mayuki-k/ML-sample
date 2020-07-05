# 結果確認どうしようか？

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from env import make7

x = make7.get_datas()

x_norm = MinManScaler().fit_transform(x)



def get_kmenas():
    kmeans = KMeans(n_cluster=2, random_state=0)
    f = kmeans.fit_predict(x_norm)
    return f