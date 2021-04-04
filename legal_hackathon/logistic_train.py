import sys
print(sys.executable)
import numpy as np


# functions for training
def tool_shuffle(X, Y):
	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	return (X[randomize], Y[randomize])

def tool_sigmoid(z):
	return np.clip(1/(1.0 + np.exp(-z)), 1e-8, 1-(1e-8))

def tool_f(X, w, b):
	return tool_sigmoid(np.matmul(X, w)+ b)

def tool_accuracy(y_predict, y_real):
	return 1-np.mean(np.abs(y_predict - y_real))

def tool_cross_entropy_loss(y_predict, y_real):
	return -np.dot(y_real, np.log(y_predict)) - np.dot((1-y_real), np.log(1-y_predict))

def tool_gradient(X, y_real, w, b):
	y_predict = tool_f(X, w, b)
	error = y_real - y_predict
	w_grad = -np.sum(error*X.T, 1)
	b_grad = -np.sum(error)
	return w_grad, b_grad

def tool_normalize(X, train = True, X_mean = None, X_std = None):
	if train:
		X_mean = np.mean(X, 0).reshape(1, -1)
		X_std = np.std(X, 0).reshape(1, -1)
	X = (X - X_mean) / (X_std + 1e-8)
	return X, X_mean, X_std

# read training data
np.random.seed(0)
X_train_path = "./data_2/X_train"
Y_train_path = "./data_2/Y_train"
if sys.argv.__len__() >= 3:
	X_train_path = sys.argv[1]
	Y_train_path = sys.argv[2]

with open(X_train_path) as fxt:
	next(fxt)
	X_train = np.array([line.strip("\n").split(",")[1:] for line in fxt], dtype=float)
	print(X_train)
with open(Y_train_path) as fyt:
	next(fyt)
	Y_train = np.array([line.strip("\n").split(",")[1] for line in fyt], dtype=float)

# Normalize
X_train, X_mean, X_std = tool_normalize(X_train, True)

# prepare for training
train_size = int(len(X_train)*(1-0.1))
X_dev = X_train[train_size:]
Y_dev = Y_train[train_size:]
X_train = X_train[:train_size]
Y_train = Y_train[:train_size]
dev_size = X_dev.shape[0]
data_dim = X_train.shape[1]

w = np.zeros((data_dim,))
b = np.zeros((1,))
max_iter = 10
batch_size = 8
learningRate = 0.2

train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# training
step = 1
for epoch in range(max_iter):
	X_train, Y_train = tool_shuffle(X_train, Y_train)
	for idx in range(int(np.floor(train_size/batch_size))):
		X = X_train[idx*batch_size:(idx+1)*batch_size]
		Y = Y_train[idx*batch_size:(idx+1)*batch_size]
		w_grad, b_grad = tool_gradient(X, Y, w, b)
		w = w - learningRate / np.sqrt(step) * w_grad
		b = b - learningRate / np.sqrt(step) * b_grad
		step += 1

	Y_train_float = tool_f(X_train, w, b)
	Y_train_predict = np.round(Y_train_float)
	train_acc.append(tool_accuracy(Y_train_predict, Y_train))
	train_loss.append(tool_cross_entropy_loss(Y_train_float, Y_train) / train_size)

	Y_dev_float = tool_f(X_dev, w, b)
	Y_dev_predict = np.round(Y_dev_float)
	dev_acc.append(tool_accuracy(Y_dev_predict, Y_dev))
	dev_loss.append(tool_cross_entropy_loss(Y_dev_float, Y_dev) / dev_size)
np.save("logistic_w.npy", w)
np.save("logistic_X_mean.npy", X_mean)
np.save("logistic_X_std.npy", X_std)
np.save("logistic_b.npy", b)


print("train loss :", train_loss[-1])
print("Dev loss :", dev_loss[-1])
print("train acc :", train_acc[-1])
print("Dev acc :", dev_acc[-1])

# plot the result
import matplotlib.pyplot as plt
plt.plot(train_loss)
plt.plot(dev_loss)
# plt.title("Loss")
# plt.legend(["train", "dev"])
# plt.savefig("loss.png")
# plt.show()

plt.plot(train_acc)
plt.plot(dev_acc)
# plt.title("Accuracy")
plt.legend(["train loss", "dev loss", "train acc", "dev acc"])
plt.savefig("acc.png")
# plt.show()
