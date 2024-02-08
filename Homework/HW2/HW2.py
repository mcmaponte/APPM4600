import numpy as np
import matplotlib.pyplot as plt

# Question 4 - Python Practice
# a) Create a vector t from 0 to pi incrementing by pi/30

t = np.linspace(0, np.pi, 31)

# Create a vector y = cos(t)
y = np.cos(t)

# Compute a sum
start = 1
end = len(t)
S = 0
print(t[1])

for i in range(start, end):
    S = S + t[i]*y[i]

print('the sum is: ', S)

# b) Wavy circles, plot parametric curves

# define parameter
theta = np.linspace(0,2*np.pi,1000)
# define constants
R = 1.2
deltaR = 0.1
f = 15
p = 0

x = R*(1+deltaR*np.sin(f*theta+p))*np.cos(theta)
y = R*(1+deltaR*np.sin(f*theta+p))*np.sin(theta)

plt.plot(x, y, 'k')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Figure 1: Parametric Curve')
plt.axis('equal')
plt.grid(True)
plt.savefig('HW2_4bi.png')

# Do it again but fudge with all the constants in a for loop

def x_ii(theta_ii, R_ii, deltaR_ii, f_ii, p_ii): return R_ii*(1+deltaR_ii*np.sin(f_ii*theta_ii+p_ii))*np.cos(theta_ii)


def y_ii(theta_ii, R_ii, deltaR_ii, f_ii, p_ii): return R_ii*(1+deltaR_ii*np.sin(f_ii*theta_ii+p_ii))*np.sin(theta_ii)

start = None
end = None

start = 1
end = 10

for j in range(start, end):
    R_ii = j
    deltaR_ii = 0.05
    f_ii = 2 + j
    p_ii = np.random.uniform(0,2)

    x_temp = x_ii(theta, R_ii, deltaR_ii, f_ii, p_ii)
    y_temp = y_ii(theta, R_ii, deltaR_ii, f_ii, p_ii)

    # do the same plot
    plt.plot(x_temp,y_temp)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Figure 2: Parametric Curves with Changing Parameters')
plt.axis('equal')
plt.grid(True)
#plt.show()
plt.savefig('HW2_4bii.png')