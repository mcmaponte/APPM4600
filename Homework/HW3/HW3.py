# Matthew Menendez-Aponte - APPM 4600 HW3

# Imports
import numpy as np
import matplotlib.pyplot as plt
from bisection_example_modified import bisection
from fixedpt_example import fixedpt

# Modified bisection_example to also return count
# 1 a) Apply Bisection Method to 2x - 1 = sin(x)
def Q1c():
    tol = 10e-8
    Nmax = 100
    def f(x): return 2*x - 1 - np.sin(x)
    a = -np.pi/2
    b = np.pi/2
    [astar, ier, Count] = bisection(f, a, b, tol, Nmax)
    print('Final approximation x = ', astar)
    print('Number of iterations used = ', Count)

def Q2():
    a = 4.82
    b = 5.2
    tol = 1e-4
    Nmax = 100
    def f_1(x): return (x-5)**9  # unexpanded
    def f_2(x): return x**9 - 45*x**8 + 900*x**7 - 10500*x**6 + 78750*x**5 - 393750*x**4 + 1312500*x**3 - 2812500*x**2 + 3515625*x - 1953125

    [xstar_unexpanded, ier_unexpanded, Count_unexpanded] = bisection(f_1, a, b, tol, Nmax)
    [xstar_expanded, ier_expanded, Count_expanded] = bisection(f_2, a, b, tol, Nmax)

    print('Q2')
    print('The final approximation for the unexpanded form is x = ', xstar_unexpanded)
    print('The iterations required for the unexpanded form is n = ', Count_unexpanded)
    print('')

    print('The final approximation for the expanded form is x = ', xstar_expanded)
    print('The iterations required for the expanded form is n = ', Count_expanded)

def Q3():
    a = 1
    b = 4
    tol = 10**(-3)
    def f(x): return x**3 + x - 4
    Nmax = 100

    [xstar, ier, Count] = bisection(f, a, b, tol, Nmax)
    print('Q3')
    print('The final approximation of x is: ', xstar)
    print('The number of iterations required to achieve tolerance is: ', Count)


def Q5():
    def f(x): return x - 4*np.sin(2*x) - 3
    # a) plot f(x)
    x = np.linspace(-2.5,7.5,1000)
    f_eval = f(x)
    plt.plot(x,f_eval,'k')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Plotting f(x) from -2.5 to 7.5')
    plt.hlines(0,-2.5,7.5,'r')
    plt.legend(['f(x)', 'y=0'])
    plt.grid(True)
    plt.savefig('5a.png')

    # b) use existing fixedpt algorithm
    tol = 0.5e-10
    Nmax = 500
    number = 4
    def f(x): return -(4/number)*np.sin(2*x) + x + (1/number)*x - (3/number)
    def g(x): return x - 4*np.sin(2*x) - 3
    def fprime(x): return -2*np.cos(2*x) + 5/4
    #def f(x): return np.sin(2*x) - (3/4)

    # work left to right
    # root 1

    c = -0.731
    [root, ier, count] = fixedpt(f, c, tol, Nmax)
    print('Q5')
    print('1st Root is x = ', root)
    print('The 1st root took n = ', count)
    print('')

    # root 2
    c = -0.5
    [root, ier, count] = fixedpt(f, c, tol, Nmax)
    print('2nd Root is x = ', root)
    print('The 2nd root took n = ', count)
    print('')

    # root 3
    a = 1.678
    b = 1.813
    c = (a + b) / 2
    [root, ier, count] = fixedpt(f, c, tol, Nmax)
    print('3rd Root is x = ', root)
    print('The 3rd root took n = ', count)
    print('')

    # root 4
    a = 3.117
    b = 3.22
    c = (a + b) / 2
    [root, ier, count] = fixedpt(f, c, tol, Nmax)
    print('4th Root is x = ', root)
    print('The 4th root took n = ', count)
    print('')

    # root 5
    a = 4.54
    b = 4.54
    c = (a + b) / 2
    [root, ier, count] = fixedpt(f, c, tol, Nmax)
    print('5th Root is x = ', root)
    print('The 5th root took n = ', count)
    print('')

    # root 6
    c = 7
    [root, ier, count] = fixedpt(f, c, tol, Nmax)
    print('6th Root is x = ', root)
    print('The 6th root took n = ', count)
    print('')

    y = []
    for element in x:
        [root,_,_] = fixedpt(f, element, tol, Nmax)
        y.append(root)
    plt.close()
    plt.axis('equal')
    plt.plot(x, y)
    plt.plot(x, g(x), 'k')
    plt.plot(x,fprime(x), 'g')
    plt.hlines(0,-2.75,7.75,'r')
    plt.hlines(1, -2.75, 7.75, 'r')
    plt.hlines(-1, -2.75, 7.75, 'r')
    plt.xlabel('x0')
    plt.ylabel('root')
    plt.ylim(-2.75, 7.75)
    plt.xlim(-2.75, 7.75)

    plt.show()


#Q1c()
#Q2()
#Q3()
Q5()