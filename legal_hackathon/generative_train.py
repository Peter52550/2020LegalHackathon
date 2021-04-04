import numpy as np
import sys

# # functions for training
def tool_shuffle(X, Y):
	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	return (X[randomize], Y[randomize])
def tool_sigmoid(z):
	return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))
def tool_f(X, w, b):
	return tool_sigmoid(np.matmul(X, w)+ b)
def tool_accuracy(y_predict, y_real):
	return (1 - np.mean(np.abs(y_predict - y_real)))
def tool_cross_entropy_loss(y_predict, y_real):
	return -np.dot(y_real, np.log(y_predict)) - np.dot((1 - y_real), np.log(1 - y_predict))
def tool_gradient(X, y_real, w, b):
	y_predict = tool_f(X, w, b)
	error = y_real - y_predict
	w_grad = -np.sum(error*X.T, 1)
	b_grad = -np.sum(error)
	return w_grad, b_grad
def tool_predict(X, w, b):
	return np.round(tool_f(X, w, b)).astype(np.int)
def tool_normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)
    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
    return X, X_mean, X_std

# # read training data
np.random.seed(0)
X_train_path = "./data/X_train"
Y_train_path = "./data/Y_train"
if sys.argv.__len__() >= 3:
	X_train_path = sys.argv[1]
	Y_train_path = sys.argv[2]

with open(X_train_path) as fxt:
	next(fxt)
	X_train = np.array([line.strip("\n").split(",")[1:] for line in fxt], dtype = float)
with open(Y_train_path) as fyt:
	next(fyt)
	Y_train = np.array([line.strip("\n").split(",")[1] for line in fyt], dtype = float)
data_dim = X_train.shape[1]

# Normalize
X_train, X_mean, X_std = tool_normalize(X_train, train = True)

# in-class preprocess

X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])

mean_0 = np.mean(X_train_0, axis = 0)
mean_1 = np.mean(X_train_1, axis = 0)

cov_0 = np.zeros((data_dim, data_dim))
cov_1 = np.zeros((data_dim, data_dim))

for x in X_train_0:
	cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
for x in X_train_1:
	cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])

u, s, v = np.linalg.svd(cov, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)

w = np.dot(inv, mean_0 - mean_1)
b = (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1)) + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])

Y_train_pred = 1 - tool_predict(X_train, w, b)
print("generative acc : {}".format(tool_accuracy(Y_train_pred, Y_train)))

np.save("generative_w.npy", w)
np.save("generative_X_mean.npy", X_mean)
np.save("generative_X_std.npy", X_std)
np.save("generative_b.npy", b)