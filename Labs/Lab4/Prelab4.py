# Matthew Menendez-Aponte APPM 4600 Lab 4 Prelab
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt


# Shamelessly copy tests from example code
def driver():
    # test functions
    f1 = lambda x: 1 + 0.5 * np.sin(x)
    # fixed point is alpha1 = 1.4987....

    f2 = lambda x: 3 + 2 * np.sin(x)
    # fixed point is alpha2 = 3.09...

    Nmax = 100
    tol = 1e-6

    # test f1 '''
    x0 = 0.0
    [xstar, x_vector, message] = fixedPoint(f1, x0, tol, Nmax)
    print('the approximate fixed point is:', xstar)
    print('f1(xstar):', f1(xstar))
    print('Error message reads:', message)
    print('Iterated vector is: ', x_vector)

    # test f2 '''
    x0 = 0.0
    [xstar, x_vector, message] = fixedPoint(f2, x0, tol, Nmax)
    print('the approximate fixed point is:', xstar)
    print('f2(xstar):', f2(xstar))
    print('Error message reads:', message)
    print('Iterated vector is: ', x_vector)


# Practice making functions
# Create subroutine that executes a fixed point algorithm and saves each iteration

def fixedPoint(f, x0, tol, Nmax):
    Nmax = Nmax - 1
    i = 0  # Initialize count
    x_N = np.zeros((Nmax + 1, 1))  # Preallocate vector to store results
    x_N[0] = x0  # 1st guess is given
    while i < Nmax:
        i += 1  # iterate count
        xi = f(x0)  # fix the point
        x_N[i] = xi  # store intermediate value

        if abs(xi - x0) < tol:  # check for tolerance
            x_final = xi
            return [x_final, x_N[0:i], 'Tolerance achieved']
        else:
            x0 = xi
    x_final = xi
    return [x_final, x_N, 'Maximum number of iterations reached']




# Exercises
# 2.2.1 figure out a method to find degree of convergence from a vector of iterations
def findConvergance(p):
    end = len(p)-1
    r = p[end]
    found = False
    alpha = 1.
    i = 0
    limit = np.zeros((100,1))
    while not found:
        limit[i] = np.mean(np.divide(abs(p[1:end - 1] - r), (abs(p[0:end - 2] - r))**alpha))
        if limit[i] < 1:
            alpha += 1
            i += 1
        else:
            found = True
            return[alpha-1, limit[i-1]]

# 2.2.2 The fixed point of g(x) = (10/x+4)^0.5 is p = 1.3652300134140976
# a) from x0 = 1.5 find iterations required to converge to 10^-10

if __name__ == '__main__':
    driver()
    def f(x): return (10/(x+4))**0.5
    x0 = 1.5
    tol = 10**-10
    Nmax = 15
    [x_final, x_N, message] = fixedPoint(f, x0, tol, Nmax)
    print('2.2.2 a) The number of iterations required is: ', len(x_N))

    # b) Use the method from 1 to estimate the order of convergence of a)

    [alpha, k] = findConvergance(x_N)
    print('2.2.2 b)')
    print('This is the degree of convergence: ', alpha)
    print('This is the constant of convergence: ', k)





