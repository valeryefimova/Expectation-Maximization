import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sys import argv

script, d_name = argv
data = pd.read_csv("dataset/" + d_name, delimiter=',', header=None)
print(data.shape)

target = 784
pr = [x for x in data.columns if x not in [target]]
X = data[pr]
y = data[target]
y = y.ravel()

#print(y)
cl = GaussianMixture(n_components=10)
cl.fit(X)
labels = cl.predict(X)

f = open('mnist_labels.txt', 'w')
f.write(str(labels))

acc = accuracy_score(y, labels)
print(acc)
f.write("\n" + str(acc))