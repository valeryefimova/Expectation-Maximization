import numpy as np
import math


class GaussianMixture:
    def __init__(self, n_components):
        self.n_components = n_components
        self.weights = [1. / n_components for _ in range(n_components)]
        self.gaussians = None

    def score_samples(self, samples):
        if np.size(samples, 1) != np.size(self.gaussians[0].mean, 1):
            raise AssertionError("The number of dimensions of the given data is not compatible with the model")

        lpr = self.compute_weighted_log_probability(samples)
        logprob = GaussianMixture.__log_sum_exp__(lpr)

        responsibilities = np.zeros((len(samples), len(self.gaussians)))
        for i in range(len(samples)):
            for j in range(len(self.gaussians)):
                responsibilities[i, j] = math.exp(lpr[i, j] - logprob[i])

        return logprob, responsibilities

    @staticmethod
    def __log_sum_exp__(data):
        lse = [0 for _ in range(len(data))]

        for i in range(len(data)):
            data_max = np.amax(data[i])

            for j in range(len(data[i])):
                lse[i] += math.exp(data[i, j] - data_max)

            lse[i] = data_max + math.log(lse[i])
        return lse

    def compute_weighted_log_probability(self, samples):
        lpr = self.log_probability(samples)
        for j in range(len(lpr[0])):
            logw = math.log(self.weights[j])
            for i in range(len(lpr)):
                lpr[i, j] += logw
        return lpr

    def log_probability(self, samples):
        nmix = len(self.gaussians)
        nsamples = np.size(samples, 0)

        log_prob = np.zeros((nsamples, nmix), dtype=list)
        for i in range(nmix):
            lp = self.gaussians[i].estimate_log_probability(samples)
            for j in range(nsamples):
                log_prob[j, i] = lp[j]
        return log_prob
