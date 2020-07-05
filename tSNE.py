from sklearn.manifold import TSNE



def get_tsne():
    tsne = TSNE(n_components=2, random_state=0)
    y_tsne = tsne.fit_transform(X)
    return y_tsne
