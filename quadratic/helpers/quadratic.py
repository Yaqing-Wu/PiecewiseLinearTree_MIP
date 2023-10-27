import numpy as np

def generate_Xy(n, n_point):
    # n: dimension of input vector; n_point: # points sampled
    # y = x^{T}Ax + bx +c; A, b, and c are randomly selected
    A = np.random.normal(0, 5, size=(n, n))
    b = np.random.normal(0, 1, size=(n, 1))
    c = np.random.normal(0, 1, size=(1, 1))[0][0]
    X = []
    for i in range(n):
        if i == 0:
            X = np.random.uniform(-2, 2, size=(n_point,1))
        else:
            X = np.append(X, np.random.uniform(-2, 2, size=(n_point,1)), axis = 1)
    y = []
    for x in X:
        y.append(np.matmul(np.matmul(x, A),x) + np.matmul(x, b)[0] + c)
    y = np.array(y)
    
    return X, y

