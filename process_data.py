import pandas as pd

d_name = "dataset/mnist_test"
data = pd.read_csv(d_name + ".csv", delimiter=',', header=None)
print(data.shape)
df_30 = data[0:30000]
df_30.to_csv(d_name + "_30.csv", index=False, header=None)
