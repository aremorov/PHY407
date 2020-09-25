"""
Author: Arya Kimiaghalam
Code for Question 1
Please run the plotting codes separately. 
"""
import numpy as np
import matplotlib.pyplot as plt
##Part-a:
def f(x):
    """
    model function.
    """
    return (np.e)**(-(x**2))

def derivative_calc(f,h,x_value,method):
    """
    calculates the numerical value of the derivative of a function
    based on the given step size.
    Method = 0 (forward difference method)
    Method = 1 (centred difference method)
    """
    if method == 0:
        return ((f(x_value+h)-f(x_value))/h)
    if method == 1:
        return ((f(x_value+h)-f(x_value-h))/(2*h))
    else:
        return "Please enter method correctly."

derivative_values = []
h_values = []
for n in np.arange(-16,1):           #generating the values of h, increasing by a factor of 10 each time.
    h_values.append((10.0)**n)
for step in h_values:
    derivative_values.append(derivative_calc(f,step,0.5,0))     #generate the derivative values and append it to the value list.

##Part-b:
def f_prime(x):
    """
    Derivative of function f, as above.
    """
    return (-2*x)*((np.e)**(-(x**2)))

analytical = f_prime(0.5) #analytical value of the derivatove at x=0.5
error = []
for value in derivative_values:
    error.append(abs(value-analytical))

##Part-c:
plt.loglog(h_values,error,c='r',linewidth=3, label = "Forward diff. method")
plt.title("The Error vs. Step Size Logarithmic Graph")
plt.xlabel("Log(h)")
plt.ylabel("Log(Error)")
"""
When h is small, the first term in eq 5.91 dominates the error, that being the
rounding error of the computer.
However in the large h values, the second term in eq 5.91 dominates, that being
the Error (approx. error) we get by estimating the function as the second order 
Taylor series of itself.
"""
##Part-d:

#derivative function has a dual mode in part a so we use that again.
derivative_values_2 = []
for step in h_values:
    derivative_values_2.append(derivative_calc(f,step,0.5,1))
error_2 = []  #error in center difference method.
for value in derivative_values_2:
    error_2.append(abs(value-analytical))
    
plt.loglog(h_values,error_2,c="blue",linewidth=2, label="Centred diff. method")
plt.legend()
"""
It can be observed that the forward and centred difference method are almost 
(with the exception of 10^-13 to 10^-11, in which 
the centred method becomes superior) the same in accuracy up to the 10^-9 mark.
However from 10^-9 to 10^-1, the centred method 
beats the forward method by a large margine. It should also be noted that
the lowest error of the centred method is reached at h = 10^-6.
"""
