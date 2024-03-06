# APPM 4600 - Lab 7 - Matthew Menendez-Aponte

# imports
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

# import useful things from provided interp.py
from interp import eval_lagrange
from interp import dividedDiffTable
from interp import evalDDpoly

# import from prelab
from Prelab7 import vandermonde
from Prelab7 import evaluatePolynomial

# Lab 7 items
# interpolating the following function with different methods


def f(x): return 1 / (1 + (10*x)**2)


def interp_nodes(interval, N):
    xj = np.linspace(interval[0], interval[1], N)
    return xj


def eval_points(interval, eval_N):
    xj = np.linspace(interval[0], interval[1], eval_N)
    return xj

# interpolate and evaluate f(x) via
#   monomial expansion
#   Lagrange Polynomials
#   Newton-Divided Differences

# Monomial expansion (Vandermonde)
# do for N = 2, 3, 4, ..., 10


def driver():
    # interpolate and evaluate f(x) via
    #   monomial expansion
    #   Lagrange Polynomials
    #   Newton-Divided Differences
    interval = np.array([-1, 1])

    # Monomial expansion (Vandermonde)
    # do for N = 2, 3, 4, ..., 10
    N = range(2,11)
    N_eval = 1000

    # preallocate y_interp to save all points
    y_interp_vandermonde = np.zeros((len(N), N_eval))

    for i in range(len(N)):
        xn = interp_nodes(interval, N[i])
        yn = f(xn)
        a = vandermonde(xn, yn)
        print(a)

        zn = eval_points(interval, N_eval)
        y_interp_vandermonde[i, :] = evaluatePolynomial(a, zn)

    # do it again for lagrange
    # same N and N_eval
    y_interp_lagrange = np.zeros((len(N), N_eval))

    for i in range(len(N)):
        xn = interp_nodes(interval, N[i])
        yn = f(xn)
        zn = eval_points(interval, N_eval)
        for j in range(len(zn)):
            y_interp_lagrange[i, j] = eval_lagrange(zn[j], xn, yn, N[i]-1)

    # do it again for Newton-Divided difference
    # same N and N_eval
    y_interp_newton = np.zeros((len(N), N_eval))

    for i in range(len(N)):
        xn = interp_nodes(interval, N[i])
        yn = np.zeros((len(xn), len(xn)))
        yn[:, 0] = f(xn)
        y = dividedDiffTable(xn,yn,N[i])
        zn = eval_points(interval, N_eval)
        for kk in range(N_eval):
            y_interp_newton[i, kk] = evalDDpoly(zn[kk], xn, y, N[i]-1)


    print('vandermonde shape: ', y_interp_vandermonde.shape)
    print('lagrange shape: ', y_interp_lagrange.shape)
    print('NewtonDD shape: ', y_interp_newton.shape)

    # Evaluate error
    truth = f(zn)
    plt.figure(1)
    plt.plot(zn, truth, 'k')
    plt.plot(zn, y_interp_vandermonde[7, :])
    plt.legend(['truth','N=2'])
    plt.show()


if __name__ == '__main__':
    driver()