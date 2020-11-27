import numpy as np
import matplotlib.pyplot as plt

def f(x): 
    return x**(-1/2) / (1+np.e**x)

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
print("Mean value integral: " + str(I_mv))
print("Importance sampling integral: " + str(I_is))

#Now run each method 100 times and create histogram of results:
mv_list = [I_mv] #append the values previously computed
is_list = [I_is]

for i in range(100):
    mv_list.append(mv_int(N,a,b,f))
    is_list.append(is_int1(N,a,b,f,p))
    

plt.hist(mv_list, 10, range=[0.8, 0.88])
plt.xlabel(r"$\int_a^b f(x) dx$")
plt.ylabel("Number of values")
plt.title("Histogram of Mean Value Method values")
plt.savefig("mv_a.pdf")

plt.hist(is_list, 10, range=[0.8, 0.88])
plt.xlabel(r"$\int_a^b f(x) dx$")
plt.ylabel("Number of values")
plt.title("Histogram of Importance Sampling Method values")
plt.savefig("mv_a.pdf")


#Part b:

def f(x): 
    return np.exp(-2*abs(x-5))

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
print("Mean value integral: " + str(I_mv))
print("Importance sampling integral: " + str(I_is))

plt.hist(mv_list, 10, range=[0.9, 1.1])
plt.xlabel(r"$\int_a^b f(x) dx$")
plt.ylabel("Number of values")
plt.title("Histogram of Mean Value Method values")
plt.savefig("mv_b.pdf")

plt.hist(is_list, 10, range=[0.9, 1.1])
plt.xlabel(r"$\int_a^b f(x) dx$")
plt.ylabel("Number of values")
plt.title("Histogram of Importance Sampling Method values")
plt.savefig("mv_b1.pdf")

plt.hist(is_list, 10, range=[0.98, 1.02])
plt.xlabel(r"$\int_a^b f(x) dx$")
plt.ylabel("Number of values")
plt.title("Histogram of Importance Sampling Method values")
plt.savefig("mv_b2.pdf")