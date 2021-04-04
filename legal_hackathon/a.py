import numpy as np

a = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(np.mean(a, 0).reshape(3,1))
print(a.shape[0])
print(a.sum(axis = 0))