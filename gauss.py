from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from env import make8

x, y = make8.get_datas()
x_norm = StandardScaler().fit_transform(x)


def get_gaussian(x):
    gmm = GaussianMixture(n_components=3, random_state=5)
    gmm.fit(x)
    return gmm

def get_probs(gmm, data):
    probs = gmm.predict_proba(data)[0]
    for idx, prob in enumerate(probs):
        print(f'class{idx} probability:{prob:.3f}')