from sklearn.cluster import KMeans

sse = []

def set_sse(x):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(x)
    sse.append(kmeans.inertia_)

#これでグラフをみる