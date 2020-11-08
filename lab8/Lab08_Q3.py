"""
Lab08 Q3
Author: Arya Kimiaghalam.
Code inspired from the slicing technique in Example 9.3, page 423 of Newman.
***************IMPORTANT NOTE*****************:
Please run the initial conditions of part a and b separately to avoid repetition.
You can do this by commenting out part a while working on part b and so forth.
"""
import numpy as np
import matplotlib.pyplot as plt

###PART-a:
##First set of constants and variables:
#Constants
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
u = np.zeros([J+1],float)  #initial velocities.
eta = H + A * np.e**(-((x-mu)**2)/sigma**2) - np.average(A * np.e**(-((x-mu)**2)/sigma**2))   #initial depths (i.e. eta).

###PART-b:
L = 1 #1 m
J = 150 #number of cells
dx = L/J #delta x, size of a single cell (m)
x = np.arange(0,L+dx,dx) #x array
g = 9.81 #gravity constant, (m/s^2)
H = 0.01 #height topography (m)
dt = 0.001 #timestep (s)
#
##constants for eta:
A=0.0002 #m
sigma = 0.1 #m
## Create arrays for velocity u and altitude eta
u = np.zeros([J+1],float)   #initial velocity array (m/s)
eta = H + A * np.e**(-((x)**2)/sigma**2) - np.average(A * np.e**(-((x)**2)/sigma**2))  #initial depth array.
alpha = (1*(8*np.pi))
x_0 = 0.5  #location of the topographic shift (m)
def topo_eta_b(x):
    """
    returns a topographical map of the bottom of the sea as an array if given an array of
    postitions.
    """
    return ((H-0.0004)/2)*(1+np.tanh(alpha*(x-x_0)))

eta_b = topo_eta_b(x)  #bottom topography.
   
def F(u,eta,eta_b):
    """
    calculates the two components of function F in the notes for given 
    eta and velocity (u) values.
    """
    a = (0.5)*(u**2)+g*eta
    b = (eta - eta_b)*u
    return [a,b]

topo_mid = 0.5*(eta_b[0:J]+eta_b[1:J+1])   #averaged topographic altitude used for the half steps in the calculation process below.
finish_time = 4  #the finishing time. Can be set to varrying values to change the outcome of the whil loop below --> (t : [0,1,2,4]).


start_time = 0  #starting time. (always zero).
while start_time<finish_time:
    new_eta = np.zeros([J+1],float)  #empty slots of altitude that need to be filled with updated values.
    new_u = np.zeros([J+1],float)    #empty slots of velocity in the x direction that need to be filled with updated values.
    
    new_eta[0] =  eta[0] - (dt/(dx))*(F(u[1],eta[1],eta_b[1])[1] - F(u[0],eta[0],eta_b[0])[1])  #Forward difference method for the left edge.
    new_eta[-1] =  eta[J] - (dt/(dx))*(F(u[J],eta[J],eta_b[J])[1] - F(u[J-1],eta[J-1],eta_b[J-1])[1])   #Backwards difference method for te right edge.


    u_mid = 0.5*(u[1:J+1]+u[0:J]) - (dt/(2*dx))*(F(u[1:J+1],eta[1:J+1],eta_b[1:J+1])[0] - F(u[0:J],eta[0:J],eta_b[0:J])[0])  #Finding the array of u in between steps (i.e. n+1/2).
    eta_mid = 0.5*(eta[1:J+1]+eta[0:J]) - (dt/(2*dx))*(F(u[1:J+1],eta[1:J+1],eta_b[1:J+1])[1] - F(u[0:J],eta[0:J],eta_b[0:J])[1])  #Finding the array of eta in between steps (i.e. n+1/2).

    new_u[1:J] = u[1:J] - (dt/dx)*(F(u_mid[1:J],eta_mid[1:J],topo_mid[1:J])[0] - F(u_mid[0:J-1],eta_mid[0:J-1],topo_mid[0:J-1])[0])  #calculating the new velocities at all non-edge steps and assigning it to the non edge values.
    new_eta[1:J] = eta[1:J] - (dt/dx)*(F(u_mid[1:J],eta_mid[1:J],topo_mid[1:J])[1] - F(u_mid[0:J-1],eta_mid[0:J-1],topo_mid[0:J-1])[1])  #calculating the new etas at all non-edge steps and assigning it to the non edge values.

    #updating our eta to the current new eta:
    eta = np.copy(new_eta)
    u = np.copy(new_u)
    start_time = start_time +dt   #updating time.


#ploting etas for desired final time:
plt.plot(x,eta)
plt.title("Wave Front at t = "+str(finish_time),fontsize=14)
plt.xlabel("X [m]",fontsize=14)
plt.ylabel("\u03B7"+" [m]",fontsize=14)


#plotting velocities:
plt.plot(x,u)
plt.title("Wave speed at t = "+str(finish_time),fontsize=14)
plt.xlabel("X [m]",fontsize=14)
plt.ylabel("Speed [m/s]",fontsize=14)




    
