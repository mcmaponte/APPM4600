import numpy as np
import matplotlib.pyplot as plt

# Barycentric lagrange interpolant

def f(x): return 1/(1 + (16*x)**2)

def p_x(xi,yi,z,w):
    n = len(xi)
    if z in xi:
        i = np.where(xi == z)[0]
        p_z = yi[i]
    else:
        num = 0.0
        den = 0.0
        for i in range(n):
            num += ((w[i]*yi[i])/(z-xi[i]))
            den += ((w[i])/(z-xi[i]))

        p_z = num/den
    return p_z

def Q1():
    # define x and f(x)
    n_nodes = 19
    n_eval = 1000

    # define points to evaluate
    zi = np.linspace(-1, 1, n_eval)

    #xi = np.linspace(-1, 1, n_nodes)
    i = np.arange(n_nodes)
    print(i)
    xi = np.cos(((2*i + 1)*np.pi)/(2*(n_nodes+1)))
    print(xi)
    yi = f(xi)

    # compute weights
    w = np.zeros(n_nodes)
    for i in range(n_nodes):
        product = 1.0
        for j in range(n_nodes):
            if i != j:
                product = product * (xi[j] - xi[i])
        w[i] = product
    w = 1/w


    interpolant = np.zeros(n_eval)
    for k in range(n_eval):
        interpolant[k] = p_x(xi, yi, zi[k], w)

    plt.figure()
    plt.plot(xi, yi, 'ok' ,markersize=10, label='nodes')
    plt.plot(zi, f(zi),'r-', label='function')
    plt.plot(zi, interpolant, 'k', label='interpolant')
    plt.plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('1. b) Plotting for n=19 and Chebyshev Nodes')
    plt.legend()
    #plt.show()
    plt.savefig('1cn19.png')





    # find weights


if __name__ == '__main__':
    Q1()