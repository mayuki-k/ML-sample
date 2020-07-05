from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
import numpy as np
from env import make10

def get_silhouette(x):
    kmeans = Kmeans(n_clusters=3, random_state=0)
    labels = kmeans.fit_predict(x)
    silhouette_vals = silhouette_samples(x, labels, metrics='euclidean')
    return silhouette_vals