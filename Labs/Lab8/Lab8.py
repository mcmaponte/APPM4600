# Lab 8 Pre-Lab
# Matthew Menendez-Aponte

# Write a subroutine that constructs and evaluates a line through two points at point alpha

import numpy as np
import matplotlib.pyplot as plt
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
        i_min = np.argmin(abs(np.floor(x - alpha[i])))
        z[i] = a[i_min] * alpha[i] + b[i_min]

    return a, b, z

if __name__ == '__main__':
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    alpha = np.linspace(0, 5, 100)

    (a, b, z) = linear_spline(x, y, alpha)

    plt.plot(alpha, z, 'k', label='Spline fit')
    #plt.plot(x, y, 'r', label='f(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


