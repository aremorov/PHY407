import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.optimize as opt

# part a) pseudo-code.

"""
1) from keyboard read the initial normalized population.
2) from keyboard read the max number of years.
3) from keyboard, read the r value.
4) set the population list to empty set.
5) append the initial normalized population to the population list.
6) for every year in the list of years, calculate (next year) = r*(1-(previous year)))*(previous year).
7) then append this value to the population list.
8) repeat-until the loop reaches the year limit.
9) plot the population list vs. the list of years.
"""
 

#Q-2-b
def population(x_0,y,r):
    """
    n_0: initial population.
    y: years.
    n_max: max population.
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
    return np.array(evo)

##Q-2-c
#plt.plot(population(0.1,50,2.2), label="r=2.2")
#plt.plot(population(0.1,50,2.5), label="r=2.5")
#plt.plot(population(0.1,50,2.8), label="r=2.8")
#plt.plot(population(0.1,50,3.1), label="r=3.1")
#plt.plot(population(0.1,50,3.4), label="r=3.4")
#plt.plot(population(0.1,50,3.7), label="r=3.7")
#plt.title("Growth of the normalized value of the population over 50 years")
#plt.xlabel("Time (year)")
#plt.ylabel("Normalized Population")
#plt.legend()
#plt.show()

##Q-2-d
#r_values = np.arange(2,4,0.015)
#i = []
#for h in range(1500):
#    i.append(h/1500)
#for j in range(len(r_values)):
#    if r_values[j] < 3:
#        plt.scatter(r_values[j]*np.ones((1,100))[0],population(0.1,2000,r_values[j])[-100:], color= (random.random(),random.random(),random.random()),s=0.1)
#    else:
#        plt.scatter(r_values[j]*np.ones((1,1000))[0],population(0.1,2000,r_values[j])[-1000:], color= (random.random(),random.random(),random.random()), s=0.1)
#    plt.title("Bifurcation Diagram")
#    plt.xlabel("Bifurcation parameter (r)")
#    plt.ylabel("Normalized Population")
##Q-2-e:
#r_values = np.arange(3.738,3.745,0.00001)
#i = []
#for h in range(1500):
#    i.append(h/1500)
#for j in range(len(r_values)):
#    if r_values[j] < 3:
#        plt.scatter(r_values[j]*np.ones((1,100))[0],population(0.1,2000,r_values[j])[-100:], color= (random.random(),random.random(),random.random()),s=0.1)
#    else:
#        plt.scatter(r_values[j]*np.ones((1,1000))[0],population(0.1,2000,r_values[j])[-1000:], color= (random.random(),random.random(),random.random()), s=0.1)
#    plt.title("Bifurcation Diagram")
#    plt.xlabel("Bifurcation parameter (r)")
#    plt.ylabel("Normalized Population")

##Q-2-f:
r = 3.1
x_0_1 = 0.1
x_0_2 = 0.1+0.1*random.random()
#plt.plot(population(x_0_1,50,r),color='r', label="Population 1")
#plt.plot(population(x_0_2,50,r),color='green', label="Population 2")
#plt.legend()
#plt.title("Normalized population over time. r=3.1")
#plt.ylabel("Normalized population")
#plt.xlabel("Time (years)")

#Q-2-g:
r = 3.1
x_0_1 = 0.1
x_0_2 = 0.1+0.1*random.random()
#plt.semilogy(abs(population(x_0_1,50,r)-population(x_0_2,50,r)),c='maroon')
plt.title("Difference in normalized population over time. r=3.1")
plt.ylabel("Difference in normalized population")
plt.xlabel("Time (years)")


def fit(x,a,l,c):
    return a*((np.e)**(l*x))+c
fff = []
for i in range(51):
    fff.append(fit(i,-0.2,-0.2,0.2))
plt.plot(fff, label="lambda = -0.2")
plt.plot(abs(population(x_0_1,50,r)-population(x_0_2,50,r)))
plt.legend()


