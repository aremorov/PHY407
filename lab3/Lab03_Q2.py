"""
Code for Q.2 in Lab03.
Author: Arya Kimiaghalam
Please run the plotting codes separately.
"""
from math import factorial as fac
import matplotlib.pyplot as plt
from numpy import array,arange,sqrt,e,pi
from gaussxw import gaussxwab   #importing Newman's code.

##Part-a:
def H(n,x):
    """
    Calculates the value of the nth Hermite polynomial
    at position x.
    n has to be a non negative integer.
    """
    H_values = []   #a list of sequential H values for different n's up to n=n.
    H_values.append(1)  #appends H_0.
    H_values.append(2*x)  #appends H_1.
    if n>1:
       for i in range(1,n):
           H_values.append((2*x*H_values[-1])-(2*i*H_values[-2]))
       return H_values[-1]
    elif n == 0:
        return H_values[0]
    else:
        return H_values[1]

def psi(n,x):
    """
    defines the value of the n-th wavefunction (i.e. stationary state) at a particular
    x-position.
    """
    a = 1/(sqrt((2**n)*fac(n)*sqrt(pi)))
    b = (e)**(-1*(x**2)*0.5)
    H_n = H(n,x)
    return a*b*(H_n)

def psi_data(n,r,N):
    """
    Computes an array of data for the nth wave function.
    n: n-value.
    r: range of x-values. For example r=5 will lead to x in [-5,5].
    N: Number of calculation points in this interval.
    """
    x_range = arange(-r,r,(2*r)/N)
    data = []
    for point in x_range:
        data.append(psi(n,point))
    return array(data)
##plotting things:
x_range = arange(-4,4,(8/100))
psi_0 = psi_data(0,4,100)
psi_1 = psi_data(1,4,100)
psi_2 = psi_data(2,4,100)
psi_3 = psi_data(3,4,100)
plt.plot(x_range,psi_0,c='k',label = "n = 0",linewidth=3)
plt.plot(x_range,psi_1,c='green',label = "n = 1",linewidth=3)
plt.plot(x_range,psi_2,c='b',label = "n = 2",linewidth=3)
plt.plot(x_range,psi_3,c='r',label = "n = 3",linewidth=3)
plt.title("Graph of Wave Functions at Different n-values", fontsize=14)
plt.xlabel("x", fontsize=14)
plt.ylabel("\u03C8"+"(n,x)",fontsize=14)
plt.legend(fontsize=12)
#Part-b:
x_range_2 = arange(-10,10,20/500)
psi_30 = psi_data(30,10,500)  #500 points.
plt.plot(x_range_2,psi_30, c="maroon",linewidth=3)
plt.xlabel("x",fontsize=14)
plt.ylabel("\u03C8"+"(n,x)",fontsize=14)
plt.title("Graph of Wave Functions at n = 30", fontsize=14)

##Part-c:

def psi_prime(n,x):
    """
    Finds the derivative in respect to position of 
    the wave function at position x.
    """
    a = 1/(sqrt((2**n)*fac(n)*sqrt(pi)))
    b = (e)**(-1*(x**2)*0.5)
    third_factor = (-1*x*H(n,x))+(2*n*H(n-1,x))
    return a*b*third_factor
   
#--------------------------------------------------------------------------.
def gauss_q_integral_calc(N,n,a,b,f):
    """
    N: number of sample points.
    a,b: interval lef and right boundary.
    f: the function we want its integral.
    *Inspired from the Textbook.
    """
    x,w = gaussxwab(N,a,b)
    s = 0.0
    for j in range(N):
        s += (w[j]*f(x[j],n))
    return s

#calculating <x2> and <p2>:
def inner_func_1(z,n):
    """
    This is the inner function of the integral definition of <x2>.
    z: changed variable goes from -pi/2 to pi/2. we previously had x = (z/(1-z**2)).
    """
    return ((psi(n,(z/(1-z**2))))**2)*((z/(1-z**2))**2)*((1+z**2)/((1-z**2)**2))

def inner_func_2(z,n):
    """
    This is the inner function of the integral definition of <p2>.
    z: changed variable goes from -pi/2 to pi/2. we previously had x = (z/(1-z**2).
    """
    return (abs(psi_prime(n,(z/(1-z**2))))**2)*((1+z**2)/((1-z**2)**2))


def results(n):
    """
    It returns the following parameters for a state n in the following order:
    expectation of x2, expectation of p2, total energy, position uncertainty, momentum uncertainty. 
    """
    exp_x_2 = gauss_q_integral_calc(100,n,-1,1,inner_func_1)
    exp_p_2 = gauss_q_integral_calc(100,n,-1,1,inner_func_2)
    E = 0.5*(exp_x_2+exp_p_2)
    sigma_x = sqrt(exp_x_2)
    sigma_p = sqrt(exp_p_2)
    return [exp_x_2,exp_p_2,E,sigma_x,sigma_p]

energy = []
e_x = []
e_p = []
x_2 = []
p_2 = []
for n in range(0,16):
    energy.append(results(n)[2])
    e_x.append(results(n)[3])
    e_p.append(results(n)[4])
    x_2.append(results(n)[0])
    p_2.append(results(n)[1])

#generating <x2> vs. n plot.
n_axis = list(range(0,16))
plt.scatter(n_axis,x_2, color='r')
plt.title(r"<$x^2$>"+" vs. n-value",fontsize=14)
plt.xlabel("n",fontsize=14)
plt.ylabel(r"<$x^2$>",fontsize=14)
#generating <p2. vs. n plot.
plt.scatter(n_axis,p_2, color='b')
plt.title(r"<$p^2$>"+" vs. n-value",fontsize=14)
plt.xlabel("n",fontsize=14)
plt.ylabel(r"<$p^2$>",fontsize=14)
#generating position uncertainty vs. n plot.
plt.scatter(n_axis,e_x, color='r')
plt.plot(n_axis,e_x, color='gray')
plt.title("Uncertainty in Position vs. n-value",fontsize=14)
plt.xlabel("n",fontsize=14)
plt.ylabel("\u03C3",fontsize=14)

#generating momentum unvcertainty vs. n plot.
plt.scatter(n_axis,e_p, color='b')
plt.plot(n_axis,e_p, color='gray')
plt.title("Uncertainty in Momentum vs. n-value", fontsize=14)
plt.xlabel("n",fontsize=14)
plt.ylabel("\u03C3",fontsize=14)
#generating position uncertainty vs. momentum uncertainty graph for comparison analysis.  
plt.scatter(e_x,e_p)
plt.plot(e_x,e_p,c='r')
plt.title("Relationship of Position and Momentum Uncertainties",fontsize=12)
plt.xlabel("Uncertainty in Position",fontsize=12)
plt.ylabel("Uncertainty in Momentum",fontsize=12)

#generating the Energy plot.
plt.scatter(n_axis,energy,color="purple")
plt.title("Energy of Different States of the system", fontsize=14)
plt.xlabel("n",fontsize=14)
plt.ylabel("Energy", fontsize=14)

