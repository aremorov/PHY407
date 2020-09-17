"""
Bifurcation and Chaos:
This code is designed to numerically produce the population of a species based on
the logistic map algorithm and to calculate the Lyapunov exponent for random shifts in initial conditions.
"""
import numpy as np
import matplotlib.pyplot as plt
import random

# Q-2-a pseudo-code.

"""
1) from keyboard read the initial normalized population.
2) from keyboard read the number of years of the evolution.
3) from keyboard, read the r value.
4) set the population list to empty set.
5) append the initial normalized population to the population list.
6) for every year in the list of years, calculate (next year) from r*(1-(previous year)))*(previous year).
7) then append this value to the population list.
8) repeat-until the loop reaches the year limit.
9) plot the population list vs. the list of years.
"""
 

#Q-2-b
def population(x_0,y,r):
    """
    x_0: initial population.
    y: years.
    r = max rate.
    returns the population of a colony after y number of years, includes initial population for t=0.
    """
    evo = []
    for year in range(y):
        if len(evo) == 0:
            evo.append(x_0)
        x_i = evo[-1]
        x_f = r*(1-x_i)*x_i
        evo.append(x_f)
    return np.array(evo)   #convert list to an array.

#Q-2-c
#plots of the normalized population over 50 years with varrying r values.
plt.plot(population(0.1,50,2.2), label="r=2.2",c='r')
plt.plot(population(0.1,50,3.2), label="r=3.2",c='g')
plt.plot(population(0.1,50,3.4), label="r=3.4",c='k')
plt.plot(population(0.1,50,3.6), label="r=3.6", c='y')
plt.title("Growth of the normalized value of the population over 50 years")
plt.xlabel("Time (year)")
plt.ylabel("Normalized Population")
plt.legend()
plt.show()

#Q-2-d
#Generating the r values with the increment specified by the lab instructions. 
r_values = np.arange(2,4,0.015)

#Plotting different x_p values for varrying r-values and finally combining them all to produce the bifurcation diagram.
for j in range(len(r_values)):
    if r_values[j] < 3:
        plt.scatter(r_values[j]*np.ones((1,100))[0],population(0.1,2000,r_values[j])[-100:], color= (random.random(),random.random(),random.random()),s=0.1)
    else:
        plt.scatter(r_values[j]*np.ones((1,1000))[0],population(0.1,2000,r_values[j])[-1000:], color= (random.random(),random.random(),random.random()), s=0.1)
    plt.title("Bifurcation Diagram")
    plt.xlabel("Bifurcation parameter (r)")
    plt.ylabel("Normalized Population")
#Q-2-e:
#Generating r-values with an increment specified in the lab instructions.
r_values = np.arange(3.738,3.745,0.00001)
#Plotting different x_p values for varrying r-values and finally combining them all to produce the bifurcation diagram.
for j in range(len(r_values)):
    if r_values[j] < 3:
        plt.scatter(r_values[j]*np.ones((1,100))[0],population(0.1,2000,r_values[j])[-100:], color= (random.random(),random.random(),random.random()),s=0.1)
    else:
        plt.scatter(r_values[j]*np.ones((1,1000))[0],population(0.1,2000,r_values[j])[-1000:], color= (random.random(),random.random(),random.random()), s=0.1)
    plt.title("Bifurcation Diagram")
    plt.xlabel("Bifurcation parameter (r)")
    plt.ylabel("Normalized Population")

#Q-2-f:
r = 3.9 #an r-value that produced chaotic behaviour.
x_0_1 = 0.1
x_0_2 = 0.1+0.000001*random.random()  #produces random change in the initial population.
plt.plot(population(x_0_1,35,r),'r', label="Population 1")
plt.plot(population(x_0_2,35,r),'g--', label="Population 2")
plt.legend()
plt.title("Normalized population over time. r=3.9")
plt.ylabel("Normalized population")
plt.xlabel("Time (years)")

#Q-2-g:
r = 3.9     #an r-value that produced chaotic behaviour.
x_0_1 = 0.1
x_0_2 = 0.1+0.000001*random.random()         #produces random change in the initial population.
plt.semilogy(abs(population(x_0_1,50,r)-population(x_0_2,50,r)),c='maroon')
plt.title("Difference in normalized population over time. r=3.9")
plt.ylabel("Difference in normalized population")
plt.xlabel("Time (years)")
#
#
#Define an exponential fit function:
def fit(x,a,l):
    return a*((np.e)**(l*x))
fit_data = []
#Generate fit values:
for i in range(25):
    fit_data.append(fit(i,0.0000007,0.57))
plt.semilogy(fit_data, label="labmda=0.57")
plt.legend()

