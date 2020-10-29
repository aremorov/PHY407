"""
Lab07 Question-1
Author: Arya Kimiaghalam
Please run plotting codes separately.
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time
g = 1
m = 10
L = 2
#Now we are going to define a function that calculates data points for the orbit.
def f(r):
    """
    This function will calculate the value of
    the right hand side of every first-order ODEs.
    r: the initial value array. [x,s,y,p].
    Here s is the first derivative of x and p is the first derivative of y.
    We do this to split the ODE into first order ODEs.

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
def mtd_1():   
    """
    Uses RK4 method with fixed interval lengths to calculate the values of
    positions and velocity.
    """
    t_i = 0
    t_f = 10
    N = 10000
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
        k1 = h*f(r)
        k2 = h*f(r+0.5*k1)
        k3 = h*f(r+0.5*k2)
        k4 = h*f(r+k3)
        r += (k1+2*k2+2*k3+k4)/6
    return [xpoints,ypoints]

#implementing the adaptive timestep method:
def rho(h,delta,e_x,e_y):
    """
    calcuates rho for every step based on epsilons and delta.
    """
    return (h*delta)/(np.sqrt(e_x**2+e_y**2))
def RK4(h,r_i):
    """
    updates r_i based on RK4 method and interval length h.
    """
    k1 = h*f(r_i)
    k2 = h*f(r_i+0.5*k1)
    k3 = h*f(r_i+0.5*k2)
    k4 = h*f(r_i+k3)
    r_f = r_i + (k1+2*k2+2*k3+k4)/6
    return r_f
delta = 10**-6
def mtd_1_adapt():
    """
    Calculates the position, velocity and time step values for the system based
    on the RK4 method and the adaptive midpoint technique.
    """
    r = np.array([1,0,0,1],float)
    xpoints = []
    spoints = []
    ypoints = []
    ppoints = []
    h_i = [0.01]
    while 2*sum(h_i) < 10:    #making sure we stop at around the right moment.
        xpoints.append(r[0])
        spoints.append(r[1])
        ypoints.append(r[2])
        ppoints.append(r[3])
        m_1 = RK4(h_i[-1],RK4(h_i[-1],r))
        m_2 = RK4(2*h_i[-1],r)
        #calculating the errors for both a and y:
        e_x = abs(m_2[0] - m_1[0])
        e_y = abs(m_2[2] - m_1[2])
        if rho(h_i[-1],delta,e_x,e_y) >= 1 and 16 >= rho(h_i[-1],delta,e_x,e_y):
            r = RK4(h_i[-1],RK4(h_i[-1],r))  #using x1 not x2.
            h_prime = h_i[-1]*(rho(h_i[-1],delta,e_x,e_y)**(1/4))
            h_i.append(h_prime)
        elif rho(h_i[-1],delta,e_x,e_y) >= 1 and 16 < rho(h_i[-1],delta,e_x,e_y):
            r = RK4(h_i[-1],RK4(h_i[-1],r))  #using x1 not x2.
            h_prime = 2*h_i[-1]
            h_i.append(h_prime)
        else:
            h_prime = h_i[-1]*(rho(h_i[-1],delta,e_x,e_y)**(1/4))
            h_i.append(h_prime)
            r = RK4(h_i[-1],RK4(h_i[-1],r))
    return [np.array(xpoints),np.array(ypoints),np.array(h_i)]
            
    
    
###plot generation of the trajectory based on the two methods:
plt.figure(figsize=(7.5,7.5))
#
plt.plot(mtd_1()[0],mtd_1()[1],'k',markersize=0.8,label="Fixed interval method")
plt.plot(mtd_1_adapt()[0],mtd_1_adapt()[1],'r.',markersize=2.5,label="Adaptive Method")
plt.title("Orbit of the Ball-Bearing around the Rod",fontsize=14)
plt.xlabel("X-position",fontsize=14)
plt.ylabel("Y-position",fontsize=14)
plt.axis("equal")
plt.legend()
#    
###Part-b:
#calculating processing time for the two methods:
t = []
s1 = time()
mtd_1_adapt()
f1 = time()
t.append(f1-s1)

s2 = time()
mtd_1()
f2 = time()
t.append(f2-s2)
print("Adaptive method time: "+str(t[0])+" s" + "; " + "Non-adaptive time: "+str(t[1])+" s")
#plotting the timesteps vs time along with the radial distance for further analysis:
plt.figure(figsize=(12,5))
plt.plot((10/len(mtd_1_adapt()[2]))*np.arange(0,len(mtd_1_adapt()[2][:-1])),(mtd_1_adapt()[1])**2+(mtd_1_adapt()[0])**2,label="Radial Position")
plt.plot((10/len(mtd_1_adapt()[2]))*np.arange(0,len(mtd_1_adapt()[2])),50*mtd_1_adapt()[2],label="Time Step x50")
plt.title("Time step and Radial Position vs. Time",fontsize=14)
plt.xlabel("Time (s)",fontsize=14)
plt.ylabel("Timestep length (s)/Radial Position",fontsize=14)
plt.legend()
