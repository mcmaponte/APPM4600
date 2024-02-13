# Matthew Menendez-Aponte, Lea Hibbard, and Beckett Hyde
# Lab 4 - 3.4 Exercises

# Pseudo code for Steffson's Method
# Inputs: fixed point function f(x), an initial guess x0, maximum number of iterations, stopping tolerance
# outputs: final guess xStar, output code
import numpy as np
from Prelab4 import findConvergance

def steffenson(f,x0,tol,Nmax):
    p_vec = np.zeros((Nmax,1))
    i = 0 # counting steps taken
    pn = x0
    p_vec[0] = pn

    while i < Nmax:
        a = pn
        b = f(pn)
        c = f(b)
        pn_new = a - (b - a) ** 2 / (c - 2 * b + a)
        p_vec[i+1] = pn_new
        if abs(pn_new - pn) < tol:  # tolerance reached
            pstar = pn_new
            return[pstar, p_vec[0:i+1], 'tolerance achieved']
        else:  # ain't it chief
            i += 1
            pn = pn_new
    pstar = pn_new
    return[pstar, p_vec[0:i+1], 'Maximum number of steps reached']

def driver_lab4():
    # do the exercises from 3.4
    def g(x): return ((10)/(x+4))**0.5
    x0 = 1.5
    tol = 10e-10
    Nmax = 1000
    [pstar, p_vec, message] = steffenson(g,x0,tol,Nmax)
    print('Found with message: ' + message)
    print('The root found is: ', pstar)
    print('Compare to the actual root of: ', 1.3652300134140976)

    # Estimate order of convergance
    print('This is p_vec', p_vec)
    [alpha, k] = findConvergance(p_vec)
    print('This is the degree of convergence: ', alpha)
    print('This is the constant of convergence: ', k)


driver_lab4()

