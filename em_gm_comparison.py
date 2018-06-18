import numpy as np
import matplotlib.pyplot as plt
from os import walk
from sklearn.decomposition import PCA

import constants
import sm

metrics = [
        constants.davies_bouldin_metric, # 1
        constants.dunn_metric, # 2
        constants.cal_har_metric,  # 3, from scikit-learn
        constants.silhouette_metric, # 4, from scikit-learn
        constants.dunn31_metric,  # 5
        constants.dunn41_metric, # 6
        constants.dunn51_metric,  # 7
        constants.dunn33_metric,  # 8
        constants.dunn43_metric, # 9
        constants.dunn53_metric,  # 10
        # Constants.gamma_metric,  # 11  # BROKEN
        constants.cs_metric, # 12
        constants.db_star_metric, # 13
        constants.sf_metric, # 14
        constants.sym_metric, # 15
        constants.cop_metric,  # 16
        constants.sv_metric, # 17
        constants.os_metric,  # 18
        constants.s_dbw_metric, # 19
        constants.c_ind_metric # 20
    ]

dir = "dataset/cl/"
for (dirpath, dirnames, filenames) in walk(dir):
    for filename in filenames:
        d_name = filename[0:-4]
        data_set = np.genfromtxt(dir + filename, delimiter=',', skip_header=0)
        np.random.shuffle(data_set)
        print(d_name)
        print(data_set.shape)

        for metric in metrics:
            smc = sm.SM(d_name, metric, data_set)
            em_val, em_par = smc.smac_em()
            gm_val, gm_par = smc.smac_gm()

            print(metric + ": EM: " + str(em_val) + ", GM: " + str(gm_val))
            print("EM clusters: " + str(em_par) + ", GM clusters: " + str(gm_par["n_components"]))

            gm_labels = smc.gm_labels(gm_par)
            em_labels = smc.em_labels(em_par)

            X = PCA(n_components=2).fit_transform(data_set)

            plt.figure(figsize=(15, 10))
            plt.subplot(221)
            plt.scatter(X[:, 0], X[:, 1], c=em_labels)
            plt.title("Predicted by EM " + d_name + ", " + metric)

            plt.subplot(222)
            plt.scatter(X[:, 0], X[:, 1], c=gm_labels)
            plt.title("Predicted by GM " + d_name  + ", " + metric)
            plt.show()
