# Lab 7 Pre-Lab

# Import things
import numpy as np
import matplotlib.pyplot as plt

# For pre-lab, write down the
# linear system that you need to solve in order to find the coefficient {aj }n
# j=0. The matrix that you
# need to invert is called the Vandermonde matrix.

def vandermonde(xn, yn):
    # initialize V matrix
    # V a_n = y_n
    n = len(xn)
    n_vec = np.arange(0, n, 1)
    V = np.zeros([n, n])

    for i in range(n):
        column = np.transpose(np.power(xn, n_vec[i]))
        V[[i]] = column

    V = np.transpose(V)
    a = np.linalg.solve(V, yn)

    return a

def evaluatePolynomial(cn,zn):
    n = len(cn)
    j = len(zn)
    sum  = np.zeros((1, j))

    for i in range(n):
        sum = sum + cn[i]*zn**i
    return sum

if __name__ == "__main__":
    #    return f_zn
    a = vandermonde(np.array([-1,-0.5,0]), np.array([1,0.75,1]))

    points = evaluatePolynomial(a,np.array([1,2.5]))
    print('points: ', points)