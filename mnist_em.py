import expectation_maximization as em
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from sys import argv

def dist(p1, p2):
    assert len(p1) == len(p2)
    res = 0
    for i in range(len(p1)):
        res += np.power((p1[i] - p2[i]), 2)

    return math.sqrt(res)

# dataset from http://yann.lecun.com/exdb/mnist/

script, d_name = argv
data_set = np.genfromtxt("dataset/" + d_name, delimiter=',', skip_header=0)
#data_set = data_set[0:100]

# X = PCA(n_components=2).fit_transform(data_set)
#
# plt.figure(figsize=(15, 10))
# plt.scatter(X[:, 0], X[:, 1], c='b')
# plt.title("MNIST data")
# plt.show()

np.random.shuffle(data_set)
print(data_set.shape)

length, col = data_set.shape
trainLength = int(length * 0.8)
testLength = length - trainLength
f = col - 1
X_train = data_set[0:trainLength, :f]
Y_train = data_set[0:trainLength, f:]
X_test = data_set[trainLength:(trainLength + testLength), :f]
Y_test = data_set[trainLength:(trainLength + testLength), f:]

start_t = time.time()
model_em = em.GaussianMixtureModelEM(10)
gmm = model_em.em(X_train)

from collections import Counter

# clusters:
sorted_points = dict([(gaussian, []) for gaussian in gmm.gaussians])

# true labels in clusters
caught_classes = dict([(gaussian, []) for gaussian in gmm.gaussians])

for i in range(0, trainLength):
    point = X_train[i]
    minDist = -1
    minGaussian = None
    for gaussian in gmm.gaussians:
        cur_dist = dist(point, gaussian.mean[0])
        if cur_dist < minDist or minDist == -1:
            minDist = cur_dist
            minGaussian = gaussian
    sorted_points[minGaussian].append(point)  # add point to predicted class
    caught_classes[minGaussian].append(Y_train[i][0])  # add point's label to predicted gaussian

#print(sorted_points)
#print(caught_classes)

# old_classes is a map from gaussian to old class label (0,1,2) from data:
old_classes = dict([(gaussian, -1) for gaussian in gmm.gaussians])

for gaussian in gmm.gaussians:
    caught = Counter(caught_classes[gaussian])

    # print(caught.most_common(1))

    if len(caught) == 0:
        old_classes[gaussian] = -1
    else:
        old_classes[gaussian] = caught.most_common(1)[0][0]  # Returns the highest occurring item

# calculate miss predicted values from test set:
missPredicted = 0
em_labels = []
for i in range(0, testLength):
    point = X_test[i]
    minDist = -1
    minGaussian = None
    for gaussian in gmm.gaussians:
        cur_dist = dist(point, gaussian.mean[0])
        if cur_dist < minDist or minDist == -1:
            minDist = cur_dist
            minGaussian = gaussian
    em_labels.append(old_classes[minGaussian] )
    if old_classes[minGaussian] != Y_test[i][0]:
        missPredicted = missPredicted + 1

end_t = time.time()

accuracy = (testLength-missPredicted)*1.0 / testLength
print ("ACCURACY:" + str(accuracy))
print("time = " + str(end_t - start_t))

# X = PCA(n_components=2).fit_transform(X_test)
#
# plt.figure(figsize=(15, 10))
# plt.subplot(221)
# plt.scatter(X[:, 0], X[:, 1], c=em_labels)
# plt.title("EM MNIST lables, obj = " + str(length) + ", acc = " + str(accuracy))
#
# plt.subplot(222)
# plt.scatter(X[:, 0], X[:, 1], c=Y_test.ravel())
# plt.title("True MNIST lables ")
# plt.show()




