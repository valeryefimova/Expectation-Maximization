import numpy as np
import sys
import math


class FullCovariance:
    def __init__(self):
        pass

    @staticmethod
    def create_gaussians(ngauss, ndims):
        return [FullMultivariateGaussian.from_ndims(ndims) for _ in range(ngauss)]

    @staticmethod
    def set_covariances(gaussians, cv):
        for mg in gaussians:
            mg.covar = cv.copy()

class FullMultivariateGaussian:
    def __init__(self, mean, covar):
        self.mean = mean
        self.covar = covar

    @staticmethod
    def from_ndims(ndims):
        return FullMultivariateGaussian(np.zeros((1, ndims)), np.identity(ndims))

    def estimate_log_probability(self, samples):
        lps = [self.__estimate_log_probability__(x) for x in samples]
        return lps

    def __estimate_log_probability__(self, sample):
        n = np.size(self.mean, 1)
        inv_covar = np.linalg.inv(self.covar)
        cov_det = np.linalg.det(self.covar)
        try:
            log_pdf_const_factor = - (math.log(math.sqrt(cov_det)) + (n / 2) / math.log(math.e, 2 * math.pi))
        except:
            log_pdf_const_factor = -1000000000

        xm = np.zeros((1, n))
        for i in range(n):
            xm[0, i] = sample[i] - self.mean[0, i]

        xmt = np.transpose(xm)
        v = (np.dot(xm, (np.dot(inv_covar, xmt))))[0, 0]
        return log_pdf_const_factor + (-0.5 * v)
