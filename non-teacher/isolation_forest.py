from sklearn.ensemble import IsolationForest

import warnings
warnings.simplefilter('ignore')

def get_isolation(x):
    isolation = IsolationForest(contamination=outlier_ratio, random_state=42)
    isolation.fit(x)
    return isolation

