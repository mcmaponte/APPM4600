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
        i_min = np.argmin(np.floor((x - alpha[i])))
        z[i] = a[i_min] * alpha[i] + b[i_min]

    return a, b, z


# Lab Work - Sko Buffs
# 3.2
def f(x): return 1 / (1+(10*x)**2)



if __name__ == '__main__':
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
    upper = 10

    a_results = np.zeros((upper-1, upper-2))
    b_results = np.zeros((upper-1, upper-2))
    z_results = np.zeros((len(alpha), upper-2))

    for i in range(2, upper):
        nodes = np.linspace(interval[0], interval[1], i)
        y_data = f(nodes)
        (a, b, z) = linear_spline(nodes, y_data, alpha)
        if i == 3:
            print(a)
        a_results[0: i-1, i-2] = a
        b_results[0: i-1, i-2] = b
        z_results[:, i-2] = z

    # Plot Results
    # f(x) vs splines
    plt.plot(alpha, f(alpha),'k',label='f(x)')
    i = 0
    while (i < (upper-2)):
        ilabel = 'splines for i = ' + str(i)
        plt.plot(alpha, z_results[:,  i], label=ilabel)
        i += 2
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()








