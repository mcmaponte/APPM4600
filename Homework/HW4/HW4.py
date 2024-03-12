# Matthew Menendez-Aponte APPM 4600 - HW 4

import numpy as np
import matplotlib.pyplot as plt
from newton_and_quasinewton_script import newton_method_nd
from newton_and_quasinewton_script import lazy_newton_method_nd
from newton_and_quasinewton_script import broyden_method_nd

from steepest_descent_script2 import steepest_descent
from steepest_descent_script2 import line_search


def Q1Driver():
    savefigs = False
    # problem 1:
    # f(x,y) = 3x^2 - y^2
    # g(x,y) = 3xy^2 - x^3 - 1

    # a) iterate with given formula
    def f(x, y): return 3 * x ** 2 - y ** 2
    def g(x, y): return 3 * x * y ** 2 - x ** 3 - 1
    n = 100
    xy_vec = np.zeros((2, n))

    # set initial state
    xy_vec[:, 0] = 1

    matrix = np.array([[1/6, 1/18], [0, 1/6]])

    for i in range(n-1):
        xy_vec[:, i+1] = xy_vec[:, i] - np.transpose(np.matmul(matrix, np.array([[f(xy_vec[0, i], xy_vec[1, i])],
                                                                                 [g(xy_vec[0, i], xy_vec[1, i])]])))

    # plot convergence
    plt.figure(1)
    plt.plot(xy_vec[0, :], '.k--')  # x
    plt.plot(xy_vec[1, :], '.r--')  # y
    plt.xlabel('n')
    plt.ylabel('value')
    plt.title('Convergence of x and y for 1a)')
    plt.legend(['x', 'y'], loc='upper right')
    plt.grid(True)
    if savefigs:
        plt.savefig('1aConvergence.png')
    else:
        plt.show()

    # plot convergence error
    xy_error_vec = abs(xy_vec - np.reshape(xy_vec[:, -1], [2, 1]))
    plt.figure(2)
    plt.semilogy(xy_error_vec[0, :], '.k--')  # x
    plt.semilogy(xy_error_vec[1, :], '.r--')  # y
    plt.xlabel('n')
    plt.ylabel('Absolute Error')
    plt.title('Convergence of x and y for 1a) - Error')
    plt.legend(['x', 'y'], loc='upper right')
    plt.grid(True)
    if savefigs:
        plt.savefig('1aError.png')
    else:
        plt.show()

    # 1c) Newton Method
    def evalF(x):
        F = np.zeros((len(x)))
        F[0] = f(x[0], x[1])
        F[1] = g(x[0], x[1])
        return F

    def evalJ(x):
        J = np.zeros((len(x), len(x)))
        J[0,0] = 6 * x[0]
        J[0,1] = -2 * x[1]
        J[1,0] = 3*x[1]**2 - 3*x[0]**2
        J[1,1] = 6* x[0]
        return J

    x0 = np.array([1, 1])
    tol = 1e-14
    nmax = 100
    (r, rn, nf, nJ) = newton_method_nd(evalF, evalJ, x0, tol, nmax, verb=False)
    print('root: ', r)
    print('rn: ', rn)
    print('nf: ', nf)
    print('nj: ', nJ)

    # Plot for convergence
    plt.figure(3)
    plt.plot(rn[:, 0], '.k--')  # x
    plt.plot(rn[:, 1], '.r--')  # y
    plt.xlabel('n')
    plt.ylabel('value')
    plt.title('Convergence of x and y for 1a)')
    plt.legend(['x', 'y'], loc='upper right')
    plt.grid(True)
    if savefigs:
        plt.savefig('1cConvergence.png')
    else:
        plt.show()

    # plot convergence error
    xy_error_vec = abs(rn - np.array([0.5, np.sqrt(3)/2]))
    plt.figure(4)
    plt.semilogy(xy_error_vec[:, 0], '.k--')  # x
    plt.semilogy(xy_error_vec[:, 1], '.r--')  # y
    plt.xlabel('n')
    plt.ylabel('Absolute Error')
    plt.title('Convergence of x and y for 1a) - Error')
    plt.legend(['x', 'y'], loc='upper right')
    plt.grid(True)
    if savefigs:
        plt.savefig('1cError.png')
    else:
        plt.show()

def Q2Driver():
    savefigs = True
    def f(x): return x[0]**2 + x[1]**2 - 4
    def g(x): return np.exp(x[0]) + x[1] - 1

    # root find with a) Newton, b) Lazy Newton, and c) Broyden
    # use initial guesses of i) [1,1], ii) [1,-1], and iii)) [0,0]

    x0_vec = np.array([[1, 1], [1, -1], [0.001, 0.001]])
    def evalF(x):
        F = np.zeros((len(x)))
        F[0] = f(x)
        F[1] = g(x)
        return F
    def evalJ(x):
        J = np.zeros((len(x),len(x)))
        J[0, 0] = 2*x[0]
        J[0, 1] = 2*x[1]
        J[1, 0] = np.exp(x[0])
        J[1, 1] = 1.
        return J

    tol = 1e-10
    nmax = 100

    # Preallocate results
    # Newton results
    newton_r = np.zeros((3, 2))
    newton_rn = np.zeros((nmax, 2, 3))
    newton_nf = np.zeros(3)
    newton_nJ = np.zeros(3)

    # Lazy Newton Results
    lazy_newton_r = np.zeros((3, 2))
    lazy_newton_rn = np.zeros((nmax, 2, 3))
    lazy_newton_nf = np.zeros(3)
    lazy_newton_nJ = np.zeros(3)

    # Broyden Results
    broyden_r = np.zeros((3, 2))
    broyden_rn = np.zeros((nmax, 2, 3))
    broyden_nf = np.zeros(3)

    for i in range(3):  # iterate over initial guesses
        # Newton
        (r, rn, nf, nJ) = newton_method_nd(evalF, evalJ, x0_vec[i, :], tol, nmax, verb=False)
        newton_r[i, :] = r
        newton_rn[0:len(rn), :, i] = rn
        newton_nf[i] = nf
        newton_nJ[i] = nJ

        # Lazy Newton
        (r, rn, nf, nJ) = lazy_newton_method_nd(evalF, evalJ, x0_vec[i, :], tol, nmax, verb=False)
        lazy_newton_r[i, :] = r
        lazy_newton_rn[0:len(rn), :, i] = rn
        lazy_newton_nf[i] = nf
        lazy_newton_nJ[i] = nJ

        # Broyden
        (r, rn, nf) = broyden_method_nd(evalF, evalJ(x0_vec[i, :]), x0_vec[i, :], tol, nmax-2, Bmat='fwd', verb=False)
        broyden_r[i, :] = r
        broyden_rn[0:len(rn), :, i] = rn
        broyden_nf[i] = nf

    # Plot all of the different methods
    plt.figure(1)  # i)
    k = 0
    # Newton
    plt.plot(newton_rn[0:int(newton_nf[k]), 0, k], '.k--', markersize=20, label='Newton x')
    plt.plot(newton_rn[0:int(newton_nf[k]), 1, k], 'xk-', markersize=20, label='Newton y')
    plt.ylim([-5, 5])
    plt.grid(True)
    # Lazy Newton
    plt.plot(lazy_newton_rn[0:int(lazy_newton_nf[k]), 0, k], '.r--', markersize=15, label='Lazy Newton x')
    plt.plot(lazy_newton_rn[0:int(lazy_newton_nf[k]), 1, k], 'xr-', markersize=15, label='Lazy Newton y')
    plt.grid(True)
    # Broyden
    plt.plot(broyden_rn[0:int(broyden_nf[k]), 0, k], '.b--', markersize=10, label='Broyden x')
    plt.plot(broyden_rn[0:int(broyden_nf[k]), 1, k], 'xb-', markersize=10, label='Broyden y')
    plt.grid(True)
    plt.legend()
    plt.title('2. i) Numerical Iterations for $x_0=[1,1]$')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    if savefigs:
        plt.savefig('2i.png')
    else:
        plt.show()

    plt.figure(2)  # ii)
    k = 1
    # Newton
    plt.plot(newton_rn[0:int(newton_nf[k]), 0, k], '.k--', markersize=20, label='Newton x')
    plt.plot(newton_rn[0:int(newton_nf[k]), 1, k], 'xk-', markersize=20, label='Newton y')
    plt.ylim([-5, 5])
    plt.grid(True)
    # Lazy Newton
    plt.plot(lazy_newton_rn[0:int(lazy_newton_nf[k]), 0, k], '.r--', markersize=15, label='Lazy Newton x')
    plt.plot(lazy_newton_rn[0:int(lazy_newton_nf[k]), 1, k], 'xr-', markersize=15, label='Lazy Newton y')
    plt.grid(True)
    # Broyden
    plt.plot(broyden_rn[0:int(broyden_nf[k]), 0, k], '.b--', markersize=10, label='Broyden x')
    plt.plot(broyden_rn[0:int(broyden_nf[k]), 1, k], 'xb-', markersize=10, label='Broyden y')
    plt.grid(True)
    plt.legend()
    plt.title('2. ii) Numerical Iterations for $x_0=[1,-1]$')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    if savefigs:
        plt.savefig('2ii.png')
    else:
        plt.show()

    plt.figure(3)  # iii)
    k = 2
    # Newton
    plt.plot(newton_rn[0:int(newton_nf[k]), 0, k], '.k--', markersize=20, label='Newton x')
    plt.plot(newton_rn[0:int(newton_nf[k]), 1, k], 'xk-', markersize=20, label='Newton y')
    plt.ylim([-5, 5])
    plt.grid(True)
    # Lazy Newton
    plt.plot(lazy_newton_rn[0:int(lazy_newton_nf[k]), 0, k], '.r--', markersize=15, label='Lazy Newton x')
    plt.plot(lazy_newton_rn[0:int(lazy_newton_nf[k]), 1, k], 'xr-', markersize=15, label='Lazy Newton y')
    plt.grid(True)
    # Broyden
    plt.plot(broyden_rn[0:int(broyden_nf[k]), 0, k], '.b--', markersize=10, label='Broyden x')
    plt.plot(broyden_rn[0:int(broyden_nf[k]), 1, k], 'xb-', markersize=10, label='Broyden y')
    plt.grid(True)
    plt.legend()
    plt.title('2. iii) Numerical Iterations for $x_0 = [0,0]$')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    if savefigs:
        plt.savefig('2iii.png')
    else:
        plt.show()

    print('Newton root x0: ', newton_r[0], ' Iterations: ', newton_nf[0])
    print('Newton root x1: ', newton_r[1], ' Iterations: ', newton_nf[1])
    print('Newton root x2: ', newton_r[2], ' Iterations: ', newton_nf[2])

    print('Lazy Newton root x0: ', lazy_newton_r[0], ' Iterations: ', lazy_newton_nf[0])
    print('Lazy Newton root x1: ', lazy_newton_r[1], ' Iterations: ', lazy_newton_nf[1])
    print('Lazy Newton root x2: ', lazy_newton_r[2], ' Iterations: ', lazy_newton_nf[2])

    print('Broyden root x0: ', broyden_r[0], ' Iterations: ', broyden_nf[0])
    print('Broyden root x1: ', broyden_r[1], ' Iterations: ', broyden_nf[1])
    print('Broyden root x2: ', broyden_r[2], ' Iterations: ', broyden_nf[2])


# Question 3
def Q3Driver():
    def f(x): return x[0] + np.cos(x[0]*x[1]*x[2]) - 1
    def g(x): return (1-x[0])**0.25 + x[1] + 0.05*x[2]**2 - 0.15*x[2] - 1
    def h(x): return - x[0]**2 - 0.2*x[1]**2 + 0.01*x[1] + x[2] - 1

    def evalF(x):
        F = np.zeros((len(x)))
        F[0] = f(x)
        F[1] = g(x)
        F[2] = h(x)
        return F
    def evalJ(q):
        x = q[0]
        y = q[1]
        z = q[2]
        J = np.zeros((len(q),len(q)))
        J[0,0] = 1 -y*z*np.sin(x*y*z)
        J[0, 1] = -x*z*np.sin(x*y*z)
        J[0, 2] = -x*y*np.sin(x*y*z)
        J[1, 0] = (-1) / (4 * (1-x)**(3/4))
        J[1, 1] = 1
        J[1, 2] = 0.1*z - 0.15
        J[2, 0] = -2*x
        J[2, 1] = -0.2*y + 0.01
        J[2, 2] = 1
        return J

    # Root find using Newton, Steepest Descent, and First Steepest descent method with a stopping tolerance of 5 × 10−2. Use the result of
    # this as the initial guess for Newton’s method.

    # Newton's Method
    x0 = np.array([0.1, 1, 1])
    tol = 10**-6
    nmax = 100

    (r, rn, nf, nJ) = newton_method_nd(evalF, evalJ, x0, tol, nmax, verb=False)
    print('Newtons Method')
    print('root: ', r)
    print('iterations: ', nf)
    print('Jacobian iterations: ', nJ)



    def q(x):
        Fun = evalF(x);
        return 0.5*(Fun[0]**2 + Fun[1]**2);

    def Gq(x):
        Jfun = evalJ(x)
        Ffun = evalF(x)
        return np.transpose(Jfun)@Ffun

    (r, rn, nf, ng) = steepest_descent(q, Gq, x0, tol, nmax, type='swolfe', verb=False)
    print('')
    print('Steepest Descent Method')
    print('root: ', r)
    print('iterations: ', nf)
    print('Gradient iterations: ', ng)


    #  First Steepest descent method with a stopping tolerance of 5 × 10−2. Use the result of
    # this as the initial guess for Newton’s method.

    tol = 5*10**-2
    (r, rn, nf, ng) = steepest_descent(q, Gq, x0, tol, nmax, type='swolfe', verb=False)
    # use r as an initial guess for newton
    print('')
    print('Combined Method')
    print('Steepest Descent Part')
    print('root (initial guess for Steepest Descent): ', r)
    print('iterations: ', nf)
    print('Gradient iterations: ', ng)


    (r, rn, nf, nJ) = newton_method_nd(evalF, evalJ, r, tol, nmax, verb=False)

    print('Newton Part')
    print('root: ', r)
    print('iterations: ', nf)
    print('Jacobian iterations: ', nJ)




if __name__ == '__main__':
    #Q1Driver()
    #Q2Driver()
    Q3Driver()