import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

def eval_legendre(n, x):
    # nth order
    # eval @ x

    # create vector of pn(x)
    p = np.zeros((n+1, 1))
    if n == 0:
        p[0] = 1
        return p
    elif n == 1:
        p[0] = 1
        p[1] = x
        return p
    else:
        p[0] = 1
        p[1] = x
        for i in range(2, n+1):
            p[i] = (1/((i-1)+1)) * (((2*(i-1)+1)*x*p[i-1]) - ((i-1)*p[i-2]))
        return p

def prelabdriver():
    # Check outputs for the same thing\
    n = 4
    x = 2

    p_obj_scipy = sc.special.legendre(n)
    p_scipy = p_obj_scipy(x)

    p_mine = eval_legendre(n, x)

    print('Scipy: ', p_scipy)
    print('Mine: ', p_mine)


if __name__ == '__main__':
    prelabdriver()

