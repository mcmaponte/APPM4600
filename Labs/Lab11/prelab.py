import numpy as np

def trapezoidal(f, a, b, N):
    """
    input: f: function to integrate
           a: lower bound
           b: upper bound
           N: number of subintervals
    output: approximate integral of f from a to b using the trapezoidal rule
    """
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return h * (y[0] + 2 * np.sum(y[1:-1]) + y[-1]) / 2

def simpson(f, a, b, N):
    """
    input: f: function to integrate
           a: lower bound
           b: upper bound
           N: number of subintervals
    output: approximate integral of f from a to b using the composite Simpson's rule
    """
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return h * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1]) / 3