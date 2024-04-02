# Lab 8 Pre-Lab
# Matthew Menendez-Aponte

# Write a subroutine that constructs and evaluates a line through two points at point alpha

import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv
from numpy.linalg import norm

def linear_spline(x, y, alpha):
    # inputs:
    #   x - vector of x values
    #   y - vector of y values where y=f(x)
    #   a - desired evaluation point

    # Outputs:
    #   a - slope of the line
    #   b - y-intercept of line
    #   z - evaluation at a

    delta_x  = np.diff(x)
    delta_y = np.diff(y)

    a = delta_y / delta_x

    b = -a * x[0:-1] + y[0:-1]

    z = np.zeros((len(alpha)))

    for i in range(len(alpha)):
        # find which line a should be evaluated in
        i_min = np.argmin(abs((x - alpha[i])))

        if x[i_min] >= alpha[i]:

            i_min = i_min - 1
            if i_min == -1:
                i_min = 0

        z[i] = a[i_min] * alpha[i] + b[i_min]

    return a, b, z


# Lab Work - Sko Buffs
# 3.2
def f(x): return 1 / (1+(10*x)**2)



def driver1():
    #x = np.linspace(0, 10, 100)
    #y = np.sin(x)
    #alpha = np.linspace(0, 5, 100)

    #(a, b, z) = linear_spline(x, y, alpha)

    #plt.plot(alpha, z, 'k', label='Spline fit')
    ##plt.plot(x, y, 'r', label='f(x)')
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.legend()
    #plt.show()
    # 3.2  evaluate f(x) for increasing numbers of splines
    interval = np.array([-1, 1])
    alpha = np.linspace(interval[0], interval[1], 250)
    upper = 30

    a_results = np.zeros((upper-1, upper-2))
    b_results = np.zeros((upper-1, upper-2))
    z_results = np.zeros((len(alpha), upper-2))

    for i in range(2, upper):

        nodes = np.linspace(interval[0], interval[1], i)
        y_data = f(nodes)
        (a, b, z) = linear_spline(nodes, y_data, alpha)
        a_results[0: i-1, i-2] = a
        b_results[0: i-1, i-2] = b
        z_results[:, i-2] = z

    # Plot Results
    # f(x) vs splines
    plt.plot(alpha, f(alpha), 'k', label='f(x)')
    i = 0
    while (i < (upper-2)):
        ilabel = 'splines for i = ' + str(i+2)
        plt.plot(alpha, z_results[:,  i], label=ilabel)
        i += 3
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Cubic Shenanigns
def create_natural_spline(yint, xint, N):
    #    create the right  hand side for the linear system
    b = np.zeros(N + 1)
    #  vector values
    h = np.zeros(N + 1)
    for i in range(1, N):
        hi = xint[i] - xint[i - 1]
        hip = xint[i + 1] - xint[i]
        b[i] = (yint[i + 1] - yint[i]) / hip - (yint[i] - yint[i - 1]) / hi
        h[i - 1] = hi
        h[i] = hip

    #  create matrix so you can solve for the M values
    # This is made by filling one row at a time
    A = np.zeros((N + 1, N + 1))

    Ainv =

    M =

    #  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = yint[j] / h[j] - h[j] * M[j] / 6
        D[j] = yint[j + 1] / h[j] - h[j] * M[j + 1] / 6
    return (M, C, D)

def driver2():
    ''' number of intervals'''
    Nint = 3
    xint = np.linspace(a, b, Nint + 1)
    yint = f(xint)

    ''' create points you want to evaluate at'''
    Neval = 100
    xeval = np.linspace(xint[0], xint[Nint], Neval + 1)

    (M, C, D) = create_natural_spline(yint, xint, Nint)

    print('M =', M)
    #    print('C =', C)
    #    print('D=', D)

    yeval = eval_cubic_spline(xeval, Neval, xint, Nint, M, C, D)

    #    print('yeval = ', yeval)

    ''' evaluate f at the evaluation points'''
    fex = f(xeval)

    nerr = norm(fex - yeval)
    print('nerr = ', nerr)

    plt.figure()
    plt.plot(xeval, fex, 'ro-', label='exact function')
    plt.plot(xeval, yeval, 'bs--', label='natural spline')
    plt.legend
    plt.show()

    err = abs(yeval - fex)
    plt.figure()
    plt.semilogy(xeval, err, 'ro--', label='absolute error')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #driver1()







