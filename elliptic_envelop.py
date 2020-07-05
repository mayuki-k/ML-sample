from sklearn.covariance import EllipticEnvelope



def get_envelope(x):
    envelope = EllipticEnvelope(contamination=0.15)
    envelope.fit(x)
    return envelope
