"""
Lab08 Q3
Author: Arya Kimiaghalam, code frame work writtern by Andrey Remorov
"""
import numpy as np
import matplotlib.pyplot as plt
from pylab import clf, plot, xlim, ylim, show, pause, draw
###PART-a:
##First set of constants and variables:
# Constants
L = 1 #1 m
J = 50 #number of cells
dx = L/J #delta x, size of a single cell (m)
x = np.arange(0,L+dx,dx) #x array
g = 9.81 #gravity constant, (m/s^2)
H = 0.01 #height topography (m)
dt = 0.01 #timestep (s)
eta_b = np.zeros([J+1]) #height topography (m)



#constants for eta:
A=0.002 #m
mu = 0.5 #m
sigma = 0.05 #m
# Create arrays for velocity u and altitude eta
u = np.zeros([J+1],float)
eta = H + A * np.e**(-((x-mu)**2)/sigma**2) - np.average(A * np.e**(-((x-mu)**2)/sigma**2))
#
####PART-b:
#L = 1 #1 m
#J = 150 #number of cells
#dx = L/J #delta x, size of a single cell (m)
#x = np.arange(0,L+dx,dx) #x array
#g = 9.81 #gravity constant, (m/s^2)
#H = 0.01 #height topography (m)
#dt = 0.001 #timestep (s)
#
##target = 1e-6   # Target accuracy
#
##constants for eta:
#A=0.0002 #m
#sigma = 0.1 #m
## Create arrays for velocity u and altitude eta
#u = np.zeros([J+1],float)
#eta = H + A * np.e**(-((x)**2)/sigma**2) - np.average(A * np.e**(-((x)**2)/sigma**2))
#alpha = (1*(8*np.pi))
#x_0 = 0.5
#def topo_eta_b(x):
#    return ((H-0.0004)/2)*(1+np.tanh(alpha*(x-x_0)))
#eta_b = np.zeros([J+1])
#eta_b_new = []
#for loc in x:
#    eta_b_new.append(topo_eta_b(loc))
    
def F(u,eta,eta_b):
    a = (0.5)*(u**2)+g*eta
    b = (eta - eta_b)*u
    return [a,b]
def u_eta(eta_i,u_i,eta_b,time):
    """
    Calculates F up to a certain time t.
    """
    
        #initialize "new" versions of these:
    eta_new = np.copy(eta_i)
    u_new = np.copy(u_i)
    eta = np.copy(eta_i)
    u = np.copy(u_i)
    
    for t in range(int(time/dt)): #simulation time of 4s
        # Calculate new values u, eta
        for i in range(J+1):
            if i==0:
                u_new[i] = 0 #forward differences
                eta_new[i] = eta[i] - (dt/(dx))*((eta[i+1]-eta_b[i+1])*u[i+1] - (eta[i]-eta_b[i])*u[i])
            if i==J:
                u_new[i] = 0 #backward differences
                eta_new[i] = eta[i] - (dt/(dx))*((eta[i]-eta_b[i])*u[i] - (eta[i-1]-eta_b[i-1])*u[i-1])
            else:
                topo_mid_right = 0.5*(eta_b[i+1]+eta_b[i])
                topo_mid_left = 0.5*(eta_b[i]+eta_b[i-1])

                u_mid_right = (0.5)*(u[i+1]+u[i]) - (dt/(2*dx))*(F(u[i+1],eta[i+1],eta_b[i+1])[0] - F(u[i],eta[i],eta_b[i])[0])
                
                eta_mid_right = (0.5)*(eta[i+1]+eta[i]) - (dt/(2*dx))*(F(u[i+1],eta[i+1],eta_b[i+1])[1] - F(u[i],eta[i],eta_b[i])[1])
                
                u_mid_left = (0.5)*(u[i-1]+u[i]) - (dt/(2*dx))*(F(u[i],eta[i],eta_b[i])[0] - F(u[i-1],eta[i-1],eta_b[i-1])[0])
                
                eta_mid_left = (0.5)*(eta[i-1]+eta[i]) - (dt/(2*dx))*(F(u[i],eta[i],eta_b[i])[1] - F(u[i-1],eta[i-1],eta_b[i-1])[1])
                
                u_new[i] = u[i] - (dt/(dx))*(F(u_mid_right,eta_mid_right,topo_mid_right)[0] - F(u_mid_left,eta_mid_left,topo_mid_left)[0])
                
                eta_new[i] = eta[i] - (dt/(dx))*(F(u_mid_right,eta_mid_right,topo_mid_right)[1] - F(u_mid_left,eta_mid_left,topo_mid_left)[1])
                
        # update the arrays to new timestep
        eta = np.copy(eta_new)
        u = np.copy(u_new)
    return [eta_new,u_new]


#for tval in np.arange(0,5.5,0.05):
#    clf() # clear the plot
#    plot(u_eta(eta,u,eta_b,tval)[0],linewidth=4) # plot the current sin curve
#    #plt.plot(eta_b_new)
#    plt.title("t = " + str(tval))
#    draw()
#    pause(0.0001) #pause to allow a smooth animation

#plt.plot(eta_b_new)
    
plt.plot(x,u_eta(eta,u,eta_b,4)[0])
plt.title("Water Wave form at t = 4s",fontsize=14)
plt.xlabel("X (m)",fontsize=14)
plt.ylabel("\u03B7"+" (m)",fontsize=14)




