from expectation_maximization import FullCovariance, GaussianMixture, KMeans
import numpy as np
import sys


class GaussianMixtureModelEM:
    def __init__(self, n_components, n_init=1, min_covar=1e-3, n_iters=100, threshold=1e-2):
        self.n_components = n_components
        self.n_init = n_init
        self.min_covar = min_covar
        self.n_iters = n_iters
        self.threshold = threshold
        self.trace = []

    def em(self, data, use_kmeans=True):
        if len(data) < self.n_components:
            raise AssertionError("GMM estimation with {} components, but got only {} samples".format(
                self.n_components, len(data)))

        gmm = GaussianMixture(self.n_components)
        max_log_prob = float("-inf")
        ctype = FullCovariance()

        for j in range(self.n_init):
            gmm.gaussians = ctype.create_gaussians(self.n_components, np.size(data, 1))

            if use_kmeans:
                k_means = KMeans(n_components=self.n_components)
                k_means.fit(data, n_iter=1000)
                mu = k_means.mu
            else:
                mu = [data[0], data[1], data[2]]

            for k in range(len(mu)):
                gmm.gaussians[k].mean[0] = mu[k]

            cv = np.cov(np.transpose(data)) + self.min_covar * np.eye(data.shape[1])
            ctype.set_covariances(gmm.gaussians, cv)

            log_likelihood = []
            bestWeights = None
            bestMixture = None

            for i in range(self.n_iters):
                (cur_log_likelihood, responsibilities) = gmm.score_samples(data)
                log_likelihood.append(sum(cur_log_likelihood))

                if i > 0 and abs(log_likelihood[i] - log_likelihood[i - 1]) < self.threshold:
                    break

                self.m_step(gmm, data, responsibilities)

                if self.n_iters > 0:
                    if log_likelihood[i] > max_log_prob:
                        max_log_prob = log_likelihood[i]
                        bestWeights = gmm.weights
                        bestMixture = gmm.gaussians

                if max_log_prob == float('inf') and self.n_iters > 0:
                    raise AssertionError("EM algorithm was never able to compute a valid likelihood given initial \
                        parameters. Try different init parameters (or increasing n_init) or \
                        check for degenerate data.")

                if self.n_iters > 0:
                    gmm.gaussians = bestMixture
                    gmm.weights = bestWeights
            self.trace = log_likelihood
        return gmm

    def m_step(self, gmm, data, responsibilities):
        weights = np.sum(responsibilities, axis=0)
        res_mat = responsibilities.copy()
        x_mat = data.copy()

        weighted_x_sum = np.dot(np.transpose(res_mat), x_mat)
        inverse_weights = [1. / (weight + 10 * sys.float_info.epsilon) for weight in weights]

        weights_sum = sum(weights)
        gmm.weights = [weight / float(weights_sum + 10 * sys.float_info.epsilon) + sys.float_info.epsilon
                       for weight in weights]

        for i in range(self.n_components):
            for j in range(len(gmm.gaussians[i].mean)):
                gmm.gaussians[i].mean[0, j] = weighted_x_sum[i, j] * inverse_weights[i]
