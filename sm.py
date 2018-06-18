from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.configspace import ConfigurationSpace
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario

from sklearn.mixture import GaussianMixture
import numpy as np

import indeces
import constants
import expectation_maximization as em
import math

class SM:
    def __init__(self, name, metric, X):
        self.metric = metric
        self.X = X
        self.name = name
        self.rows, self.cols = X.shape

    def dist(self, p1, p2):
        assert len(p1) == len(p2)
        res = 0
        for i in range(len(p1)):
            res += np.power((p1[i] - p2[i]), 2)

        return math.sqrt(res)

    def run_gm(self, cfg):
        cl = GaussianMixture(**cfg)
        cl.fit(self.X)
        labels = cl.predict(self.X)
        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)
        value = indeces.metric(self.X, n_clusters, labels, self.metric)
        return value

    def run_em(self, cfg):
        model_em = em.GaussianMixtureModelEM(**cfg)
        gmm = model_em.em(self.X)

        # clusters:
        sorted_points = dict([(gaussian, []) for gaussian in gmm.gaussians])

        label_map = dict()
        k = 0
        for gaussian in gmm.gaussians:
            label_map[gaussian] = k
            k += 1
        #print(label_map)

        labelsl = []
        for i in range(0, self.rows):
            point = self.X[i]
            minDist = -1
            minGaussian = None
            for gaussian in gmm.gaussians:
                cur_dist = self.dist(point, gaussian.mean[0])
                if cur_dist < minDist or minDist == -1:
                    minDist = cur_dist
                    minGaussian = gaussian
            sorted_points[minGaussian].append(point)  # add point to predicted class
            labelsl.append(label_map[minGaussian])

        #print(str(self.rows) + " == " + str(len(labelsl)))
        labels = np.array(labelsl)
        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)
        value = indeces.metric(self.X, n_clusters, labels, self.metric)
        return value


    def smac_gm(self):
        clu_cs = ConfigurationSpace()
        cov_t = CategoricalHyperparameter("covariance_type", ["full", "tied", "diag", "spherical"])
        tol = UniformFloatHyperparameter("tol", 1e-6, 0.1)
        reg_c = UniformFloatHyperparameter("reg_covar", 1e-10, 0.1)
        n_com = UniformIntegerHyperparameter("n_components", 2, 10)
        max_iter = UniformIntegerHyperparameter("max_iter", 10, 1000)
        clu_cs.add_hyperparameters([cov_t, tol, reg_c, n_com, max_iter])

        clu_scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                                 # "runcount-limit": Constants.num_eval,  # maximum function evaluations
                                 "cs": clu_cs,  # configuration space
                                 "deterministic": "true",
                                 "tuner-timeout": constants.timeout,
                                 "wallclock_limit": constants.timeout,
                                 "cutoff_time": constants.timeout
                                 })
        print('Run GM SMAC ' + self.name)
        smac = SMAC(scenario=clu_scenario, tae_runner=self.run_gm)
        parameters = smac.optimize()
        value = smac.get_runhistory().get_cost(parameters)
        return value, parameters

    def smac_em(self):
        clu_cs = ConfigurationSpace()
        n_init = UniformIntegerHyperparameter("n_init", 1, 15)
        n_com = UniformIntegerHyperparameter("n_components", 2, 10)
        reg_c = UniformFloatHyperparameter("min_covar", 1e-6, 0.1)
        max_iter = UniformIntegerHyperparameter("n_iters", 10, 1000)
        tr = UniformFloatHyperparameter("threshold", 1e-6, 0.1)
        clu_cs.add_hyperparameters([n_init, tr, reg_c, n_com, max_iter])

        clu_scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                                 # "runcount-limit": Constants.num_eval,  # maximum function evaluations
                                 "cs": clu_cs,  # configuration space
                                 "deterministic": "true",
                                 "tuner-timeout": constants.em_timeout,
                                 "wallclock_limit": constants.em_timeout,
                                 "cutoff_time": constants.em_timeout,
                                 "runcount-limit": 1
                                 })

        print('Run EM SMAC ' + self.name)
        smac = SMAC(scenario=clu_scenario, tae_runner=self.run_em)
        parameters = smac.optimize()
        value = smac.get_runhistory().get_cost(parameters)
        return value, parameters

    def em_labels(self, n_cl):
        model_em = em.GaussianMixtureModelEM(n_cl)
        gmm = model_em.em(self.X)

        # clusters:
        sorted_points = dict([(gaussian, []) for gaussian in gmm.gaussians])

        label_map = dict()
        k = 0
        for gaussian in gmm.gaussians:
            label_map[gaussian] = k
            k += 1

        labels = []
        for i in range(0, self.rows):
            point = self.X[i]
            minDist = -1
            minGaussian = None
            for gaussian in gmm.gaussians:
                cur_dist = self.dist(point, gaussian.mean[0])
                if cur_dist < minDist or minDist == -1:
                    minDist = cur_dist
                    minGaussian = gaussian
            sorted_points[minGaussian].append(point)  # add point to predicted class
            labels.append(label_map[minGaussian])

        return labels

    def gm_labels(self, cfg):
        cl = GaussianMixture(**cfg)
        cl.fit(self.X)
        labels = cl.predict(self.X)
        return labels