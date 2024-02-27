# Lab 6

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy.linalg import inv



# Build Slacker Newton
# Copy and paste lazy Newton from example code
def MatthewsSlackerNewton(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    ''' Slacker Newton condition, recompute J-1 every other iteration'''

    xlist = np.zeros((Nmax+1, len(x0)))
    xlist[0] = x0

    J = evalJ(x0)
    Jinv = inv(J)

    for its in range(Nmax):
        F = evalF(x0)
        x1 = x0 - Jinv*F
        xlist[its+1]=x1

        if (norm(x1-x0) < tol*norm(x0)):  # tolerance of convergence
            xstar = x1
            ier =0
            return[xstar, xlist, ier, its]

        # Slacker Newton Condition
        if its%2 == 0:  # every other
            J = evalJ(x1)
            Jinv = inv(J)

        x0 = x1

    xstar = x1
    ier = 1
    return[xstar, xlist, ier, its]

# remake evalF and evalJ for functions in the lab doc
def evalF(x):
    F = np.zeros(2)
    F[0] = 4 * x[0]**2 + x[1]**2-4
    F[1] = x[0] - x[1] - np.sin(x[0]-x[1])
    return F

def evalJ(x):
    J = np.array([[8*x[0], 2*x[1]], [1-np.cos(x[0]-x[1]), 1+np.cos(x[0]-x[1])]])
    return J

