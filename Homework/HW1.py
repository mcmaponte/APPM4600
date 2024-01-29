import numpy as np
import matplotlib.pyplot as plt



# Problem 1 - Evaluate Polynomial
x = np.arange(1.92,2.08,0.001)

# a) Evaluate via Coefficients

p_1_coeff = lambda x: x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512
y_coefficients_nolambda = x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512
y_coeff = p_1_coeff(x)


# b) Evaluate via expression
p_1_exp = lambda x: (x-2)**9
y_exp = p_1_exp(x)
y_exp_nolambda = (x-2)**9
# plot and compare


plt.plot(x,y_coeff,'')
plt.plot(x,y_exp,'')
plt.xlabel("x")
plt.ylabel("p(x)")
plt.grid(True)
plt.legend(["Expanded Polynomial with Coefficients","Expression"])
plt.title("Polynomial p(x) Calculated 2 Ways")
plt.savefig("Question1.png")
plt.show()

plt.title('No Lambda Func')
plt.plot(x,y_coefficients_nolambda,'r')
plt.plot(x,y_exp_nolambda+(0.1*10**-10),'k')
plt.legend(['Coefficients','Factored'])
plt.show()

# Problem 5
# b)
def bad_cos(x,d): return np.cos(x-d) - np.cos(x)
def good_cos(x,d): return -2 * np.sin((2*x+d)/2) * np.sin(d/2)

n = 17
delta = np.ones(n)*float(10)**np.arange(-16, n-16, 1)

x_pi = np.pi
x_big = 1*10**6

# evaluate each expression for big and small x and the different deltas

bad_cos_pix = bad_cos(x_pi, delta)
bad_cos_bigx = bad_cos(x_big, delta)

good_cos_pix = good_cos(x_pi, delta)
good_cos_bigx = good_cos(x_big, delta)

# find the difference between good and bad expressions
diff_xpi = bad_cos_pix - good_cos_pix
diff_xbig = bad_cos_bigx - good_cos_bigx

# plot the differences

plt.plot(delta,diff_xpi)
plt.plot(delta,diff_xbig)
plt.xlabel("Delta")
plt.ylabel("Difference")
plt.xscale("log")
plt.title('f with Subtraction - f without Subtraction')
plt.legend(["x1 = pi","x2=10^6"])
plt.savefig("5b.png")

# c)
def special_algorithm(x,d): return -d * np.sin(x) - ((d**2)/2)*np.cos((2*x+d)/2)

special_xpi = special_algorithm(x_pi,delta)
special_xbig = special_algorithm(x_big,delta)

# compare to other methods

# compare to method with subtraction
diff_xpi_special_vs_sub = special_xpi - bad_cos_pix
diff_xpi_special_vs_add = special_xpi - good_cos_pix

diff_xbig_special_vs_sub = special_xbig - bad_cos_bigx
diff_xbig_special_vs_add = special_xbig - good_cos_bigx

plt.close()

# Plot
plt.plot(delta,diff_xpi_special_vs_sub,'')
plt.plot(delta,diff_xpi_special_vs_add)
plt.plot(delta,diff_xbig_special_vs_sub)
plt.plot(delta,diff_xbig_special_vs_add)
plt.legend(["Taylor vs Subtraction: x=pi","Taylor vs No Subtraction: x=pi","Taylor vs Subtraction: x=10^6","Taylor vs No Subtraction: x=10^6"])
plt.title('Comparing A Taylor Approximation to the Methods from Part B')
plt.xlabel('delta')
plt.ylabel('Difference')
plt.xscale("log")
#plt.show()
plt.savefig('5c.png')
print(diff_xpi_special_vs_sub)
print(diff_xpi_special_vs_sub.shape)
print(delta.shape)