# coding: utf-8
# linear_regression/regression.py
import numpy as np
import matplotlib as plt
import time

# compute execution time
def exeTime(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        return back, time.time() - t0
    return newFunc

# h(x)
def h(theta, x):
    return (theta.T*x)[0,0]

# cost function
def J(theta, X, y):
    m = len(X)
    return (X*theta-y).T*(X*theta-y)/(2*m)

# gradient descent optimizer function
@exeTime
def gradientDescentOptimizer(rate, maxLoop, epsilon, X, y):
    """
    Args:
    rate: learning rate
    maxLoop: maximum iteration number
    epsilon: precision

    Returns:
        (theta, errors, thetas), timeConsumed
    """
    m,n = X.shape
    # initialize theta
    theta = np.zeros((n,1))
    count = 0
    converged = False
    error = float('inf')
    errors = []
    thetas = {}
    for j in range(n):
        thetas[j] = [theta[j,0]]
    while count<=maxLoop:
        if(converged):
            break
        count = count + 1
        for j in range(n):
            ############################################
            # YOUR CODE HERE!
            # deriv = ??
            deriv = -(X*theta-y).T*X[:, j]/m
            ############################################
            theta[j,0] = theta[j,0]+rate*deriv
            thetas[j].append(theta[j,0])
        error = J(theta, X, y)
        errors.append(error[0,0])
        if(error < epsilon):
            converged = True
    return theta,errors,thetas

# standarize function
def standarize(X):
    m, n = X.shape
    # normalize each feature
    for j in range(n):
        features = X[:,j]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0)
        if std != 0:
            X[:, j] = (features-meanVal)/std
        else:
            X[:, j] = 0
    return X
