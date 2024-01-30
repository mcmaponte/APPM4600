# This is a comment from Lab2, practicing using git 
# This is also a comment from Lab2, now I need to switch back to APPM4600

import numpy as np
import matplotlib.pyplot as plt


# Lab 1 - Matthew Menendez-Aponte

x=[1,2,3] #list, not an array

y = np.array([1,2,3]) #An array, we can do stuff to it

print('This is 3y',3*y)


#3.1.3 Plotting
X = np.linspace(0,2*np.pi,100)
Ya = np.sin(X)
Yb = np.cos(X)

plt.plot(X,Ya)
plt.plot(X,Yb)
plt.xlabel('X')
plt.ylabel('Y')
plt.show(block=True)
#plt.savefig("Plot1.png")

del x,y
# 3.2 - The Basics

x = np.linspace(0,10,11)
y = np.arange(0,11,1)

print('The first three entries of x are',x[[0,1,2]])


