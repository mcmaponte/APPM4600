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

def interp_nodes_chebyshev(interval, N):
    xj = np.zeros((1, N))
    for j in range(N):
        print(j)
        xj[0, j] = np.cos(np.pi - (j/(N-1)) * np.pi)
    xj = ((interval[1] - interval[0]) * ((xj+1)/2)) + interval[0]
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
    N_start = 2
    N_end = 10
    N = np.arange(N_start, N_end+1, 1)
    N_eval = 1000

    # preallocate y_interp to save all points
    y_interp_vandermonde = np.zeros((len(N), N_eval))

    for i in range(len(N)):
        xn = interp_nodes(interval, N[i])
        yn = f(xn)
        a = vandermonde(xn, yn)

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

    print('zn shape: ', zn.shape)

    # Evaluate error
    truth = f(zn)
    truth_matrix = np.tile(np.transpose(truth), (len(N), 1))
    print('truth shape', truth_matrix.shape)

    error_vandermonde = abs(y_interp_vandermonde-truth_matrix)
    error_lagrange = abs(y_interp_lagrange - truth_matrix)
    error_newton = abs(y_interp_newton - truth_matrix)

    #log_error_vandermonde = np.log10(error_vandermonde)
    #log_error_lagrange = np.log10(error_lagrange)
    #log_error_newton = np.log10(error_newton)


    # Plot Vandermonde
    plt.figure(1)
    plt.plot(zn, truth, 'k', linewidth=2, label='Truth')
    for i in range(len(N)):
        plt.plot(zn, y_interp_vandermonde[i, :], label=f'N = {i+2}')
    plt.ylim([-0.1, 1.25])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x) interpolated using vandermonde polynomials')
    plt.grid(True)
    plt.legend()
    plt.show()


    plt.figure(2) # Lagrange
    plt.plot(zn, truth, 'k', linewidth=2, label='Truth')
    for i in range(len(N)):
        plt.plot(zn, y_interp_lagrange[i, :], label=f'N = {i + 2}')
    plt.ylim([-0.1, 1.25])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x) interpolated using lagrange polynomials')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(3)  # Newton DD
    plt.plot(zn, truth, 'k', linewidth=2, label='Truth')
    for i in range(len(N)):
        plt.plot(zn, y_interp_newton[i, :], label=f'N = {i + 2}')
    plt.ylim([-0.1, 1.25])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x) interpolated using Newton polynomials')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot Errors
    # only plot error for N = 10
    plt.figure(4)
    plt.semilogy(zn, error_vandermonde[8, :], '-k', label='Vandermonde')
    plt.semilogy(zn, error_lagrange[8, :], '-b', label='Lagrange')
    plt.semilogy(zn, error_newton[8, :], '-r', label='Newton DD')
    plt.xlabel('x')
    plt.ylabel('Log10 error')
    plt.title('Error of p(x) at N=10')
    plt.legend()
    plt.show()

    # Replacing Plots
    n = 10
    plt.figure(5)
    plt.plot(zn, truth, 'k', label = 'f(x)')
    plt.plot(zn, y_interp_vandermonde[n-2, :], label='p(x) Vandermonde')
    plt.plot(zn, y_interp_lagrange[n-2, :], label='p(x) Lagrange')
    plt.plot(zn, y_interp_newton[n-2, :], label='p(x) Newton DD')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'N={n} f(x) vs p(x)')
    plt.legend()
    plt.savefig('N10.png')

    n = 6
    plt.figure(6)
    plt.semilogy(zn, error_vandermonde[n-2, :], label='Vandermonde')
    plt.semilogy(zn, error_lagrange[n-2, :], label='Lagrange')
    plt.semilogy(zn, error_newton[n-2, :], label='Newton DD')
    plt.xlabel('x')
    plt.ylabel('Log10 error')
    plt.title('Error of p(x) at N=6')
    plt.legend()
    plt.savefig('errN6.png')


def driver32():
    N = 11

    # Run for most stable method - Lagrange
    xn = interp_nodes_chebyshev(np.array([-1, 1]), N)
    yn = f(xn)

    # call lagrange
    # do it again for lagrange
    # same N and N_eval
    interval = np.array([-1,1])
    N_eval = 1000
    zn = eval_points(interval, N_eval)
    y_interp_lagrange_chebyshev = np.zeros((len(N), N_eval))

    for i in range(len(N)):
        xn = interp_nodes_chebyshev(interval, N[i])
        yn = f(xn)
        zn = eval_points(interval, N_eval)
        for j in range(len(zn)):
            y_interp_lagrange_chebyshev[i, j] = eval_lagrange(zn[j], xn, yn, N[i] - 1)



if __name__ == '__main__':
    #driver()
    driver32()
