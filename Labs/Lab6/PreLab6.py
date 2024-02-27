# 2/26/2024 - APPM4600 Prelab6
# Matthew Menendez-Aponte

# imports
import numpy as np
import matplotlib.pyplot as plt

# make functions for forward and centered difference
def forwardDifference(f,s,h):
    fprime = (f(s+h) - f(s))/h
    return fprime
def centeredDifference(f,s,h):
    fprime = (f(s + h) - f(s-h)) / (2*h)
    return fprime
def prelab6_driver():
    hVec = 0.01 * 2. ** (-np.arange(0, 10))
    print('h vector: ', hVec)

    def f(x): return np.cos(x)
    s = np.pi/8

    f_prime_forward = forwardDifference(f, s, hVec)
    f_prime_centered = centeredDifference(f, s, hVec)

    print('Approximate derivative using forward difference')
    print(f_prime_forward)
    print('Approximate derivative using centered difference')
    print(f_prime_centered)

    f_prime_true = -np.sin(s)

    log_forward_error = np.log10(abs(f_prime_forward - f_prime_true))
    log_centered_error = np.log10(abs(f_prime_centered - f_prime_true))

    log_h_vec = np.log10(hVec)

    m_forward, b_forward = np.polyfit(log_h_vec, log_forward_error, 1)
    m_centered, b_centered = np.polyfit(log_h_vec, log_centered_error, 1)
    print(f'forward slope: {m_forward:.2f}')
    print(f'centered slope: {m_centered: .2f}')

    # Plot differences
    # plot -log(h) vs log(error)
    plt.plot(log_h_vec, log_forward_error, 'xk--', markersize=10)
    plt.plot(log_h_vec, log_centered_error, '.b--', markersize=10)
    plt.legend([f'Forward Difference with Order: {m_forward:.4f}', f'Centered Difference with Order: {m_centered: .4f}'])
    plt.xlabel('-log(h)')
    plt.ylabel('log(error)')
    plt.title(f'Approximating the Derivative of cos(x) using \nForward and Centered Difference\n s = {s:.3f}')
    plt.grid(True)
    plt.gca().invert_xaxis()
#    plt.show()
    plt.savefig('Approx3.png',bbox_inches='tight')


if __name__ == '__main__':
    prelab6_driver()