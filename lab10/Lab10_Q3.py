#Lab 9, Question 1
#Author: Andrey Remorov

#In this question, I analyze the mean value and importance sampling techniques
#in numerical integration. I outputted histograms of the results of the same
#folder as this python file. The program also prints out values obtained when
#computing the integral using mean value, importance sampling for both (a),(b).

import numpy as np
import matplotlib.pyplot as plt

#integrand:
def f(x): 
    return x**(-1/2) / (1+np.e**x)

#weighting function
def p(x):
    return 1/(2*np.sqrt(x))

def mv_int(N,a,b,f): #mean value integral 
    k = 0 # will contain the total
    for i in range(N):
        x = (b-a)*np.random.random()
        k += f(x)
    return k * (b-a) / N

def is_int1(N,a,b,f,p): #importance sampling integral, for part (a)
    k2 = 0 #sum used for importance sampling
    for i in range(N):
        x = (b-a)*np.random.random()
        x2 = x**2 #value fed into non-uniform list of random numbers
        k2 += f(x2)/p(x2)
    return k2/N
        
N = 10000
a = 0.
b = 1.

I_mv = mv_int(N,a,b,f)
I_is = is_int1(N,a,b,f,p)
print("Part (a) Mean value integral: " + str(I_mv))
print("Part (a) Importance sampling integral: " + str(I_is))

#Now run each method 100 times and create histogram of results:
mv_list = [I_mv] #append the values previously computed
is_list = [I_is]

for i in range(100):
    mv_list.append(mv_int(N,a,b,f))
    is_list.append(is_int1(N,a,b,f,p))
    
#Plotting histograms:
plt.hist(mv_list, 10, range=[0.8, 0.88])
plt.xlabel(r"$\int_0^1 \frac{x^{-1/2}}{1+e^x} dx$")
plt.ylabel("Number of values")
plt.title("Histogram of Mean Value Method values (a)")
plt.tight_layout()
plt.savefig("mv_a.pdf")
plt.clf()

plt.hist(is_list, 10, range=[0.8, 0.88])
plt.xlabel(r"$\int_0^1 \frac{x^{-1/2}}{1+e^x} dx$")
plt.ylabel("Number of values")
plt.title("Histogram of Importance Sampling Method values (a)")
plt.tight_layout()
plt.savefig("is_a.pdf")
plt.clf()


#Part b:
#Here I had to compare the mean value and importance sampling techniques for a 
#different integrand:

#different function:
def f(x): 
    return np.exp(-2*abs(x-5))

#different weighting function:
def p(x):
    return (1/np.sqrt(2*np.pi))*np.exp(-(x-5)**2 / 2)


def is_int2(N,a,b,f,p): #importance sampling integral
    k2 = 0 #sum used for importance sampling
    for i in range(N):
        x = np.random.normal(loc=5) #generates a gaussian with mean 5, sd=1
        k2 += f(x)/p(x)
    return k2/N

N = 10000
a = 0.
b = 10.

I_mv = mv_int(N,a,b,f)
I_is = is_int2(N,a,b,f,p)
print("Part (b) Mean value integral: " + str(I_mv))
print("Part (b) Importance sampling integral: " + str(I_is))

#Now run each method 100 times and create histogram of results:
mv_list = [I_mv] #append the values previously computed
is_list = [I_is]

for i in range(100):
    mv_list.append(mv_int(N,a,b,f))
    is_list.append(is_int2(N,a,b,f,p))


#Plotting histograms:
plt.hist(mv_list, 10, range=[0.9, 1.1])
plt.xlabel(r"$\int_0^{10} exp(-2|x-5|) dx$")
plt.ylabel("Number of values")
plt.title("Histogram of Mean Value Method values (b)")
plt.tight_layout()
plt.savefig("mv_b.pdf")
plt.clf()

plt.hist(is_list, 10, range=[0.9, 1.1])
plt.xlabel(r"$\int_0^{10} exp(-2|x-5|) dx$")
plt.ylabel("Number of values")
plt.title("Histogram of Importance Sampling Method values (b)")
plt.tight_layout()
plt.savefig("is_b.pdf")
plt.clf()
