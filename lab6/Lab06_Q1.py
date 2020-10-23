"""
Lab06
Question-1
Author: Arya Kimiaghalam
"""
import numpy as np
import matplotlib.pyplot as plt
g = 1
m = 10
L = 2
#Now we are going to define a function that calculates data points for the orbit.
def f(r,t):
    """
    This function will calculate the value of
    the right hand side of every first-order ODEs.
    r: the initial value array. [x,s,y,p].
    Here s is the first derivative of x and p is the first derivative of y.
    We do this to split the ODE into first order ODEs.
    t: time.
    """
    #broken down into four first order ODEs.
    x = r[0]
    s = r[1]
    y = r[2]
    p = r[3]
    fx = s
    fs = -g*m*((x/(x**2+y**2))*np.sqrt((x**2+y**2)+(L**2)/4))
    fy = p
    fp = -g*m*((y/(x**2+y**2))*np.sqrt((x**2+y**2)+(L**2)/4))
    return np.array([fx,fs,fy,fp],float)
#setting our variables:
t_i = 0
t_f = 10
N = 1000
h = (t_f-t_i)/N
tpoints = np.arange(t_i,t_f,h)
xpoints = []
spoints = []
ypoints = []
ppoints = []
r = np.array([1,0,0,1],float) #initial condition of [x,s,y,p].
for t in tpoints:  #generates the points for x-pos,y-pos,x-vel and y-vel (x-vel and y-vel are called s & p here).
    xpoints.append(r[0])
    spoints.append(r[1])
    ypoints.append(r[2])
    ppoints.append(r[3])
    #RK4 method implemented:
    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6
#plot generation:
plt.figure(figsize=(4.5,4.5))
plt.plot(xpoints,ypoints,linewidth=1,c='green')
plt.title("Orbit of the Ball-Bearing around the Rod",fontsize=14)
plt.xlabel("X-position",fontsize=14)
plt.ylabel("Y-position",fontsize=14)
plt.axis("equal")

    
