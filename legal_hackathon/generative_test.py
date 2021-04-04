import numpy as np
import sys

def tool_sigmoid(z):
	return np.clip(1/(1.0 + np.exp(-z)), 1e-8, 1-(1e-8))
def tool_f(X, w, b):
	return tool_sigmoid(np.matmul(X, w)+ b)
def tool_predict(X, w, b):
	return np.round(tool_f(X, w, b)).astype(np.int)
def tool_normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
	if specified_column == None:
		specified_column = np.arange(X.shape[1])
	if train:
		X_mean = np.mean(X ,0).reshape(1, -1)
		X_std  = np.std(X, 0).reshape(1, -1)

	X = (X - X_mean) / (X_std + 1e-8)
	 
	return X, X_mean, X_std

x_test_path = "./data/X_test"
output_path = "./output_{}.csv"
if sys.argv.__len__() >= 3:
	x_test_path = sys.argv[1]
	output_path = sys.argv[2]
w = np.load("generative_w.npy")
b = np.load("generative_b.npy")
X_mean = np.load("generative_X_mean.npy")
X_std = np.load("generative_X_std.npy")
with open(x_test_path) as ft:
	features = np.array(ft.readline().strip("\n").split(",")[1:])
	X_test =  np.array([line.strip("\n").split(",")[1:] for line in ft], dtype=float)

X_test, _, _= tool_normalize(X_test, train = False, X_mean = X_mean, X_std = X_std)

Y_test_predict = 1 - tool_predict(X_test, w, b)
with open(output_path.format("generative"), "w") as fout:
	fout.write("id,label\n")
	for id, label in enumerate(Y_test_predict):
		fout.write("{},{}\n".format(id, label))

ind = np.argsort(np.abs(w))[::-1]
for i in ind[0:10]:
	print(features[i], ":", w[i])