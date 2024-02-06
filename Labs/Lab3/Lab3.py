# APPM 4600 Lab 3
# Matthew Menendez-Aponte and Lea Hibbard

# libraries
import numpy as np
from bisection_example import bisection

# Excercises
# 1.  Consider f(x)=x^2(x-1) and use bisection over the following intervals
f = lambda x: x**2 * (x-1)
Nmax = 100
tol = 1e-3

# a) (a, b) = (0.5, 2)
[astar_a, ier_a] = bisection(f,0.5,2,tol, Nmax)
print('a_root', astar_a)
print('error', ier_a)

# b) (a, b) = (-1, 0.5)
[astar_b, ier_b] = bisection(f,-1,0.5,tol, Nmax)
print('b_root', astar_b)
print('error', ier_b)

# c) (a, b) = (-1, 2)
[astar_c, ier_c] = bisection(f,-1,2,tol,Nmax)
print('c_root', astar_c)
print('error', ier_c)

# d) (a, b) = (-0.1, 0.1)
[astar_d, ier_d] = bisection(f,-0.1,0.1,tol,Nmax)
print('d_root', astar_d)
print('error', ier_d)


# What happens for each choice of interval
# for the interval in a), the bisection method is able to approximately find the root at x=1 without error.
# For the interval in b), the bisection method is not able to find the root at x=0. It returns an error saying it
# cannot tell if there is a root in the interval. The bisection method fails when the endpoints have the same sign.
# For the interval in c), the bisection method is able to find the root at x=1 without error.


# 2. Apply bisection to some functions listed below. You should set your desired accuracy to ϵ =
# 10−5
tol2 = 10e-5
Nmax2 = 100

# a) f (x) = (x − 1)(x − 3)(x − 5) with a = 0 and b = 2.4.
f2a = lambda x: (x - 1)*(x - 3)*(x - 5)
[astar_2a, ier_2a] = bisection(f2a,0,2.4,tol2,Nmax2)
print(' ')
print('2a')
print('astar_2a', astar_2a)
print('error', ier_2a)

# b) f (x) = (x − 1)2 (x − 3) with a = 0 and b = 2.
f2b = lambda x: (x-1)**2*(x-3)
[astar_2b, ier_2b] = bisection(f2b,0,2,tol2,Nmax2)
print(' ')
print('2b')
print('astar_2b', astar_2b)
print('error', ier_2b)

# c) f (x) = sin(x) with a = 0, b = 0.1. What about a = 0.5 and b = 3π/4
f2c = lambda x: np.sin(x)
[astar_2c, ier_2c] = bisection(f2c,0,0.1,tol2,Nmax2)
print(' ')
print('2c')
print('astar_2c', astar_2c)
print('error', ier_2c)



