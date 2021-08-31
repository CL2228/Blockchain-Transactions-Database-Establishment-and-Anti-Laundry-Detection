import math
import numpy as np
from  sklearn.linear_model._logistic import LogisticRegression
from scipy.linalg import expm
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

if __name__ == "__main__":
    a = np.random.random([5, 12])
    b = np.random.random([3, 12])
    c = np.random.random([23, 12])
    d = np.concatenate([a, b, c], axis=0)
    print(d.shape)