import math
import numpy as np
from  sklearn.linear_model._logistic import LogisticRegression
from scipy.linalg import expm

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

if __name__ == "__main__":
    a = expm(np.array([[1, 1, 0],[ 0, 0, 2],[ 0, 0, -1]]))
    print(a)
    print(a * 0.1)