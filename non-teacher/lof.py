from sklearn.neighbors import LocalOutlierFactor

def get_lof():
    f = LocalOutlierFactor(n_neighbors=35, contamination=outlier_ratio)
    f.fit(x)
    return f