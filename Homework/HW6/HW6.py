import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy
def Q2():
    # Question 2 code
    # Construct a composite trapezoidal rule and composite Simpson's Rule
    n = 20  # points
    a = -5.0
    b = 5.0
    def f(x): return 1 / (1 + x**2)
    def f_2(x): return ((8*x**2)/((1+x**2)**3)) - ((2)/((1+x**2)**2))
    def f_4(x): return (24*(5*(x**2 - 2)*x**2 + 1))/((x**2 + 1)**5)

    def Composite_Trapezoidal(n, f, a, b):
        n += 1
        x_vec = np.linspace(a, b, n)  # Create linearly spaced x vector
        f_vec = f(x_vec)  # Create f_vector

        h = (b-a) / (n-1)  # gap between measurments

        # do it
        approximate = (h/2) * (f_vec[0] + f_vec[-1] + 2*np.sum(f_vec[1:-1]))
        error_x = np.linspace(a, b, 1001)
        error_bound = ((h ** 3) / 12) * max(abs(f_2(error_x)))
        return approximate, error_bound

    approximate, error_bound_trap = Composite_Trapezoidal(n, f, a, b)
    print('2. a)')
    print('N = ', n)
    print('Composite Trapezoidal approximate: ', approximate)
    print('Composite Trapezoidal error bound: ', error_bound_trap)

    def Composite_Simpsons(n, f, a, b):
        if n % 2 == 0:
            points = n+1
            x_vec = np.linspace(a, b, points)
            f_vec = f(x_vec)

            h = (b-a)/n
            approximate = 0.0
            approximate2 = 0.0
            X0F = f_vec[0] + f_vec[-1]
            XODD = 0
            XEVEN = 0
            for i in range(1, points-1):
                if i % 2 == 0:  # even
                    XEVEN += f_vec[i]
                elif i % 2 == 1:  # odd
                    XODD += f_vec[i]
                approximate2 = (h/3) * (X0F + 2 * XEVEN + 4*XODD)
            error_x = np.linspace(a, b, 1001)
            error_bound = ((h**5)/90) * max(abs(f_4(error_x)))

            return approximate2, error_bound

        else:
            print('n must be even')
            return None

    approximate_Simpsons, error_bound_simp = Composite_Simpsons(n, f, a, b)
    print('Simpsons Rule yields: ', approximate_Simpsons)
    print('Simpsons Error bound: ', error_bound_simp)

    def findNForTrapError(f, a, b, tol):
        # f is f''
        error_x = np.linspace(a, b, 1001)
        calculated_error = 1.0
        n = 1
        while True:
            if calculated_error < tol:
                break
            else:
                h = (b-a)/n
                calculated_error = ((h ** 3) / 12) * max(abs(f(error_x)))
                n += 1
        return n

    def findNForSimpError(f, a, b, tol):
        # f is f''
        error_x = np.linspace(a, b, 1001)
        calculated_error = 1.0
        n = 1
        while True:
            if calculated_error < tol:
                break
            else:
                h = (b-a)/n
                calculated_error = ((h ** 5) / 90) * max(abs(f(error_x)))
                n += 1
        return n

    print('')
    print('2. b)')
    n_trap_req = findNForTrapError(f_2, a, b, 10**-4)
    n_simp_req = findNForSimpError(f_4, a, b, 10**-4)

    n_trap_req_6 = findNForTrapError(f_2, a, b, 10 ** -6)
    n_simp_req_6 = findNForSimpError(f_4, a, b, 10 ** -6)


    approximate_Trapezoidal_b, error_bound_trap = Composite_Trapezoidal(n_trap_req, f, a, b)
    approximate_Simpsons_b, error_bound_simp = Composite_Simpsons(n_simp_req, f, a, b)
    approximate_Trapezoidal_b_6, error_bound_trap_6 = Composite_Trapezoidal(n_trap_req_6, f, a, b)
    approximate_Simpsons_b_6, error_bound_simp_6 = Composite_Simpsons(n_simp_req_6+1, f, a, b)
    truth_value = 2*np.arctan(5)

    print('Truth Value: ', truth_value)
    print('')
    print('nReq for Trapezoidal with tol 10e-4: ', n_trap_req)
    print('Value of Tn: ', approximate_Trapezoidal_b)
    print('nReq for Simpsons with tol 10e-4: ', n_simp_req)
    print('Value of Sn: ', approximate_Simpsons_b)
    print('Actual Error for Trapezoidal with 120 terms: ', abs(approximate_Trapezoidal_b - truth_value))
    print('Actual Error for Simpsons with 50 terms: ', abs(approximate_Simpsons_b - truth_value))
    print('')
    print('nReq for Trapezoidal with tol 10e-6: ', n_trap_req_6)
    print('Value of Tn: ', approximate_Trapezoidal_b_6)
    print('nReq for Simpsons with tol 10e-6: ', n_simp_req_6)
    print('Value of Sn: ', approximate_Simpsons_b_6)
    print('Actual Error for Trapezoidal: ', abs(approximate_Trapezoidal_b_6 - truth_value))
    print('Actual Error for Simpsons: ', abs(approximate_Simpsons_b_6 - truth_value))



    # part c
    print('')
    print('2. c)')

    # use scipy quad and compare to Tn and Sn
    (integral_106, abserr6, info_6) = integrate.quad(f, a, b, full_output=True, epsabs=(10**(-6)))
    print('Result from scipy quad tol=10e-6: ', integral_106)
    print('Approx error from scipy quad tol=10e-6: ', abserr6)
    print('Actual error from scipy quad tol=10e-6: ', abs(truth_value - integral_106))
    print('Number of points used in scipy quad tol=10e-6: ', info_6['neval'])
    print('')

    (integral_104, abserr4, info_4) = integrate.quad(f, a, b, full_output=True, epsabs=(10 ** (-4)))
    print('Result from scipy quad tol=10e-4: ', integral_104)
    print('Approx error from scipy quad tol=10e-4: ', abserr4)
    print('Actual error from scipy quad tol=10e-4: ', abs(truth_value - integral_104))
    print('Number of points used in scipy quad tol=10e-4: ', info_4['neval'])

def Q4():
    # The gamma function
    t_vec = np.array([2, 4, 6, 8, 10])
    gamma_compare = scipy.special.gamma(t_vec)
    print('gamma_compare: ', gamma_compare)
    def integrand(x, a): return x**(a-1) * np.exp(-x)
    x_vec = np.linspace(0,25,1000)
    f_a6_vec = integrand(x_vec, 6)
    plt.figure()
    plt.plot(x_vec, f_a6_vec,'k',label='integrand t=6')
    plt.vlines(6-1, 0, 22,'r', label='x = t - 1')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plotting the Integrand of the Gamma Function')
    plt.legend()
    plt.savefig('4a.png')

    def Composite_Trapezoidal_4(n, t, a, b):
        def f(x): return x**(t-1) * np.exp(-x)
        n += 1
        x_vec = np.linspace(a, b, n)  # Create linearly spaced x vector
        f_vec = f(x_vec)  # Create f_vector

        h = (b-a) / (n-1)  # gap between measurments

        # do it
        approximate = (h/2) * (f_vec[0] + f_vec[-1] + 2*np.sum(f_vec[1:-1]))

        return approximate

    gamma_approx = np.zeros(len(t_vec))
    n_vec = np.zeros(len(t_vec))
    end = 10
    for i in range(len(t_vec)):
        n = 40*end*(t_vec[i]-1)
        n_vec[i] = n
        gamma_approx[i] = Composite_Trapezoidal_4(n, t_vec[i], 0, end*(t_vec[i]-1))

    print('gamma_approx: ', gamma_approx)
    print('n function calls: ', n_vec)
    rel_error = abs(gamma_approx-gamma_compare)/gamma_compare
    print('Rel. Error: ', rel_error)

    # c) Gauss Laguerre
    x_vec, weights = np.polynomial.laguerre.laggauss(100)
    Gauss_approx = np.zeros(len(t_vec))
    for i in range(len(t_vec)):
        f_vec = x_vec**(t_vec[i]-1)
        temp = np.multiply(f_vec, weights)
        Gauss_approx[i] = np.sum(temp)

    # evaluate rel. error
    rel_error_final = abs(Gauss_approx - gamma_compare)/gamma_compare
    print('')
    print('4.  c)')
    print('Gauss-Laguerre approximation: ', Gauss_approx)
    print('Rel. Error: ', rel_error_final)


if __name__ == '__main__':
    #Q2()
    Q4()
