import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila

from time import perf_counter_ns as timer


def driver():
    ''' create  matrix for testing different ways of solving a square
     linear system'''

    '''' N = size of system'''
    N = 100

    ''' Right hand side'''
    b = np.random.rand(N, 1)
    A = np.random.rand(N, N)

    x = scila.solve(A, b)

    test = np.matmul(A, x)
    r = la.norm(test - b)

    print('resultant norm via .solve: ', r)

    # Exercise 1
    x_LU = LU_Soln(A, b)
    test_LU = np.matmul(A, x_LU)
    r_LU = la.norm(test_LU - b)

    print('resultant norm via .LU: ', r_LU)

    ''' Create an ill-conditioned rectangular matrix '''
    N = 10
    M = 5
    A = create_rect(N, M)
    b = np.random.rand(N, 1)

    # Exercise 2
    N_vec = np.array([100, 500, 1000, 2000, 4000, 5000])

    time_regular_solve = np.zeros(len(N_vec))
    time_LU_decomp = np.zeros(len(N_vec))
    time_LU_solve = np.zeros(len(N_vec))

    for i in range(len(N_vec)):
        A = np.random.rand(N_vec[i], N_vec[i])
        b = np.random.rand(N_vec[i], 1)

        # time regular solve
        time_solve_start = timer()
        x = scila.solve(A, b)
        time_solve_end = timer()
        time_regular_solve[i] = time_solve_end - time_solve_start

        # time LU decomp
        time_LUdecomp_start = timer()
        lu, piv = scila.lu_factor(A)
        time_LUdecomp_mid = timer()
        x = scila.lu_solve((lu, piv), b)
        time_LUdecomp_end = timer()
        time_LU_decomp[i] = time_LUdecomp_mid - time_LUdecomp_start
        time_LU_solve[i] = time_LUdecomp_end - time_LUdecomp_end

    plt.figure(1)
    plt.plot(N_vec, time_regular_solve, label='Normal Solve')
    plt.plot(N_vec, time_LU_decomp + time_LU_solve, label='LU Solve + LU Factorization')
    # plt.plot(N_vec, time_LU_solve, label='LU Solve')
    plt.xlabel('N')
    plt.ylabel('Time (ns)')
    plt.title('Comparing LU Solve to Regular Solve')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Exercise 3
    # determine how many times we need to solve for LU to be faster
    N = 1000  # Choose fixed N
    n_bs = 500  # Choose how many b vecs to analyze
    A = np.random.rand(N,N)
    b_vec = np.random.rand(N,n_bs)

    # do the Lu decomp
    time_11 = timer()
    lu, piv = scila.lu_factor(A)
    time_12 = timer()
    time_important = time_12 - time_11
    print('time_important =', time_important)

    time_spent_regular_solve = np.zeros(n_bs)
    time_spent_LU_solve = np.zeros(n_bs)
    time_spent_LU_solve[0] = time_important


    # loop over b's
    for i in range(n_bs):
         time0 = timer()
         # print(np.shape(b_vec[:, i]))
         x = scila.lu_solve((lu, piv), b_vec[:, i])
         time1 = timer()
         if i == 0:
              time_spent_LU_solve[i] = time1 - time0
         else:
              time_spent_LU_solve[i] = time_spent_LU_solve[i-1] + (time1-time0)

         time2 = timer()
         x = scila.solve(A, b_vec[:, i])
         time3 = timer()
         if i == 0:
              time_spent_regular_solve[i] += time3 - time2
         else:
              time_spent_regular_solve[i] = time_spent_regular_solve[i - 1] + (time3 - time2)

    plt.figure(2)
    plt.semilogy(time_spent_regular_solve, label='Regular Solve')
    plt.semilogy(time_spent_LU_solve, label='LU Decompy + LU Solve')
    plt.legend()
    plt.grid(True)
    plt.xlabel('n bs solved')
    plt.ylabel('Time (ns)')
    plt.title('Comparing Times')
    plt.savefig('')

    print('Time regular 1: ', time_spent_regular_solve[0])




def create_rect(N, M):
    ''' this subroutine creates an ill-conditioned rectangular matrix'''
    a = np.linspace(1, 10, M)
    d = 10 ** (-a)

    D2 = np.zeros((N, M))
    for j in range(0, M):
        D2[j, j] = d[j]

    '''' create matrices needed to manufacture the low rank matrix'''
    A = np.random.rand(N, N)
    Q1, R = la.qr(A)
    test = np.matmul(Q1, R)
    A = np.random.rand(M, M)
    Q2, R = la.qr(A)
    test = np.matmul(Q2, R)

    B = np.matmul(Q1, D2)
    B = np.matmul(B, Q2)
    return B


def LU_Soln(a, b):
    # simply use scipy LU
    lu, piv = scila.lu_factor(a)
    x = scila.lu_solve((lu, piv), b)
    return x


if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()
