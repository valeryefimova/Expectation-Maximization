import numpy as np
from sklearn.metrics import euclidean_distances


class KMeans(object):
    def __init__(self, n_components=2, epsilon=0.01):
        self.n_components = n_components
        self.mu = []
        self.epsilon = epsilon

    def _get_random_point(self, X):
        return X[np.random.choice(X.shape[0], 1)[0]]

    def get_cluster_means(self, X, y):
        labs = np.arange(self.n_components)
        mu = [
            np.mean(X[np.where(y == lab)], axis=0) if np.array(np.where(y == lab)).size > 0 else self._get_random_point(
                X) for lab in labs]
        return mu

    def update_encoder(self, X, mu):
        self.y = [np.argmin(euclidean_distances(x.reshape(1, -1), mu)) for x in X]
        return self.y

    def fit(self, X, y=None, n_iter=10):
        n = X.shape[0]
        labs = np.arange(self.n_components)
        self.y = np.random.choice(self.n_components, n)
        self.cost = []
        trace = []
        for i in range(n_iter):
            self.share = np.array([np.array(np.where(self.y == lab)).size for lab in labs])
            mu = self.get_cluster_means(X, self.y)
            trace.append(mu)
            self.cost.append(np.sum([(euclidean_distances(x.reshape(1, -1), mu)).min() for x in X]))
            self.y = self.update_encoder(X, mu)
            if i > 1 and np.abs(self.cost[-1] - self.cost[-2]) < self.epsilon:
                break
        self.y = np.array(self.y)
        self.mu = mu
        self.trace = np.array(trace)
        return self

    def predict(self, X):
        return np.array(self.update_encoder(X, self.mu))