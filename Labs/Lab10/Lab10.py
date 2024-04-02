import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad


def eval_legendre(n, x):
    # nth order
    # eval @ x

    # create vector of pn(x)
    p = np.zeros((n+1, 1))
    if n == 0:
        p[0] = 1
        return p
    elif n == 1:
        p[0] = 1
        p[1] = x
        return p
    else:
        p[0] = 1
        p[1] = x
        for i in range(2, n+1):
            p[i] = (1/((i-1)+1)) * (((2*(i-1)+1)*x*p[i-1]) - ((i-1)*p[i-2]))
        return p

def prelabdriver():
    # Check outputs for the same thing\
    n = 4
    x = 2

    p_obj_scipy = sc.special.legendre(n)
    p_scipy = p_obj_scipy(x)

    p_mine = eval_legendre(n, x)

    print('Scipy: ', p_scipy)
    print('Mine: ', p_mine)


def eval_coeff(n, num, den, range):
    return quad(num, range[0], range[1], args=(n,)) / quad(den, range[0], range[1], args=(n,))

def Problem1Driver():
    def f(x):
        return x
    def w(x):
        return 1





def eval_legendre_expansion(f, a, b, w, n, x):
    # This subroutine evaluates the Legendre expansion
    # Evaluate all the Legendre polynomials at x that are needed
    # by calling your code from prelab
    # initialize the sum to 0
    pval = 0.0
    for j in range(0, n+1):
        def phi_j(x):
            p = eval_legendre(j, x)
            return p[-1]
        # define numerator
        def num_integrand(x):
            return f(x) * phi_j(x) * w(x)
        # define denominator
        def den_integrand(x):
            return phi_j(x) ** 2 * w(x)


        # use the quad function from scipy to evaluate normalizations
        denominator, err = quad(den_integrand, a, b)
        # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
        # use the quad function from scipy to evaluate coeffs
        numerator, err = quad(num_integrand, a, b)

        aj = numerator / denominator
        # accumulate into pval
        pval = pval + aj*phi_j(x)
    return pval

def driver():
#function you want to approximate
    # Problem 2
    # f = lambda x: math.exp(x)
    # Problem 3
    def f(x): return 1 / (1 + x**2)
    # Interval of interest
    a = -1
    b = 1
    # weight function
    w = lambda x: 1.
    # order of approximation
    n = 2
    # Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a, b, N+1)
    pval = np.zeros(N+1)

    for kk in range(N+1):
        pval[kk] = eval_legendre_expansion(f, a, b, w, n, xeval[kk])
    ''' create vector with exact values'''
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
    plt.figure(1)
    plt.plot(xeval, fex,'ro-', label= 'f(x)')
    plt.plot(xeval, pval,'bs--',label= 'Expansion')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('3) f(x) vs Polynomial Approximation')
    plt.grid(True)
    plt.savefig('3Curves.png')
    plt.figure(2)
    err_l = abs(pval-fex)
    plt.semilogy(xeval,err_l,'ro--',label='error')
    plt.xlabel('x')
    plt.ylabel('Log Error')
    plt.title('3) Error between Function and Approximation')
    plt.grid(True)
    plt.legend()
    plt.savefig('3Error.png')

if __name__ == '__main__':
    driver()

