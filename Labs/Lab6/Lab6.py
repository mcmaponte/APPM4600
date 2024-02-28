# Lab 6

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy.linalg import inv



# Build Slacker Newton
# Copy and paste lazy Newton from example code
def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    xlist = np.zeros((Nmax+1,len(x0)));
    xlist[0] = x0;

    for its in range(Nmax):
       J = evalJ(x0);
       F = evalF(x0);

       x1 = x0 - np.linalg.solve(J,F);
       xlist[its+1]=x1;

       if (norm(x1-x0) < tol*norm(x0)):
           xstar = x1
           ier =0
           return[xstar, xlist,ier, its];

       x0 = x1

    xstar = x1
    ier = 1
    return[xstar,xlist,ier,its];

def MatthewsSlackerNewton(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    ''' Slacker Newton condition, recompute J-1 every other iteration'''

    xlist = np.zeros(((Nmax+1), len(x0)))
    xlist[0] = x0

    J = evalJ(x0)
    Jinv = inv(J)
    SlackerCount = 0

    for its in range(Nmax):
        # Slacker Newton Condition
        if (its+1) % 2 == 0:  # every other
            print(its)
            J = evalJ(x0)
            Jinv = inv(J)
            SlackerCount += 1

        F = evalF(x0)
        x1 = x0 - np.matmul(Jinv, F)
        xlist[its+1] = x1

        if (norm(x1-x0) < tol*norm(x0)):  # tolerance of convergence
            xstar = x1
            ier =0
            return[xstar, xlist, ier, its, SlackerCount]

        x0 = x1

    xstar = x1
    ier = 1
    return[xstar, xlist, ier, its]

# remake evalF and evalJ for functions in the lab doc
def evalF(x):
    F = np.zeros(2)
    F[0] = 4 * x[0]**2 + x[1]**2-4
    F[1] = x[0] + x[1] - np.sin(x[0]-x[1])
    return F

def evalJ(x):
    J = np.array([[8*x[0], 2*x[1]], [1-np.cos(x[0]-x[1]), 1+np.cos(x[0]-x[1])]])
    return J

def slackerdriver():
    x0 = np.array([1, 0])
    tol = 1 * 10**-10.
    Nmax = 10

    # Run Newton for comparisson
    [xstar, xlist, ier, its] = Newton(x0, tol, Nmax)
    print('Newton xstar: ', xstar)
    # print('Newton xlist: ', xlist)
    print('Newton error message: ', ier)
    print('Newton iteration number: ', its)

    [xstar, xlist, ier, its, slackerCount] = MatthewsSlackerNewton(x0, tol, Nmax)
    print(' ')
    print('Slacker xstar: ', xstar)
    #print('Slacker xlist: ', xlist)
    print('Slacker error message: ', ier)
    print('Slacker iteration number: ', its)
    print('Slacker Jinv updates: ', slackerCount)

    [xstar, xlist, ier, its] = Newton(x0, tol, Nmax)

# Modify Prelab finite difference code to work with multivariable functions
def centeredDifference(f,s,h):
    fprime = (f(s + h) - f(s-h)) / (2*h)
    return fprime

def f(x):
    return np.array([[4*x[0]**2 + x[1]**2 - 4],[x[0]+x[1]-np.sin(x[0]-x[1])]])

def evalJcenteredDifference(f,x,h):
    J = np.array([[centeredDifference(f, x, [h, 0]), centeredDifference(f, x, [0, h])]])
    return J

def BeckettApproxevalJ(evalF, x, h):
    n = len(x)
    m = len(evalF(x))
    J = np.zeros((m,n))
    for i in range(n):
        xhp = x.copy()
        xhm = x.copy()
        xhp[i] = xhp[i] + h
        xhm[i] = xhm[i] - h
        J[:,i] = (evalF(xhp) - evalF(xhm))/(2*h)
    return J

def ApproxJhardcode(x, h):
    J = np.zeros((2, 2))
    J[0, 0] = (((4*(x[0]+h)**2 + x[1]**2 - 4) - ((4*(x[0]-h)**2 + x[1]**2 - 4))) / (2*h))
    J[1, 0] = (((x[0]+h) + x[1] - np.sin((x[0]+h)-x[1])) - (((x[0]-h) + x[1] - np.sin((x[0]-h)-x[1])))) / (2*h)
    J[0, 1] = (((4*x[0]**2 + (x[1]+h)**2 - 4) - ((4*x[0]**2 + (x[1]-h)**2 - 4))) / (2*h))
    J[1, 1] = ((x[0] + (x[1]+h) - np.sin(x[0]-(x[1]+h))) - ((x[0] + (x[1]-h) - np.sin(x[0]-(x[1]-h))))) / (2*h)
    return J

def MatthewHybridSlackerNewton(x0, tol, Nmax):
    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    ''' Slacker Newton condition, recompute J-1 every other iteration'''

    xlist = np.zeros(((Nmax+1), len(x0)))
    xlist[0] = x0

    h = 10**-3. * norm(x0)
    J = ApproxJhardcode(x0,h )
    Jinv = inv(J)
    SlackerCount = 0

    for its in range(Nmax):
        # Slacker Newton Condition
        if (its+1) % 2 == 0:  # every other
            print(its)
            h = 0.5**(SlackerCount+1) * (10**-3.) * norm(x0)
            J = ApproxJhardcode(x0, h)
            Jinv = inv(J)
            SlackerCount += 1

        F = evalF(x0)
        x1 = x0 - np.matmul(Jinv, F)
        xlist[its+1] = x1

        if (norm(x1-x0) < tol*norm(x0)):  # tolerance of convergence
            xstar = x1
            ier =0
            return[xstar, xlist, ier, its, SlackerCount]

        x0 = x1

    xstar = x1
    ier = 1
    return[xstar, xlist, ier, its]

def HybridSlackerDriver():
    x0 = np.array([1,0])
    tol = 1 * 10 ** -10.
    Nmax = 10
    [xstar, xlist, ier, its, SlackerCount] = MatthewHybridSlackerNewton(x0, tol, Nmax)
    print('Hybrid Slacker xstar: ', xstar)
    # print('Slacker xlist: ', xlist)
    print('Hybrid Slacker error message: ', ier)
    print('Hybrid Slacker iteration number: ', its)
    print('Hybrid Slacker Jinv updates: ', SlackerCount)



if __name__ == '__main__':
    #slackerdriver()
    HybridSlackerDriver()

