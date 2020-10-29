"""
Lab07 Question-2
Author: Arya Kimiaghalam
CREDIT:Code inspired from the code: "Bulirsch.py" from example 8.7 of 
Computational Physics by Mark Newman.
The use of this code was Authorized by the problem itself.
Please run cases with different initial conditions separately.
"""
from numpy import empty,array,arange
import matplotlib.pyplot as plt
import numpy as np

G = 6.6738*(10**-11) * ( 8760 * 60 * 60) ** 2  #time units converted into year.(from seconds)
Au = 1.495978707*(10**11) #astronomical unit.
M = 1.9891*(10**(30))

delta = 1000     # Required position accuracy per unit time.
#First set of astronomical values (Earth):
H = 1/52  #in years.
x_0 = 1.4710 * 10 ** 11  # in meters
vx_0 = 0.0
y_0 = 0.0
vy_0 = 3.0287 * 10 ** 4 * 8760 * 60 * 60  # meter/yrear
period = 1  #years.
#Second set of astronomical values (Pluto):
H = 100/52  #100 weeks looks good.
x_0 = 4.4368 * 10 ** 12  # in meters
vx_0 = 0.0
y_0 = 0.0
vy_0 = 6.1218 * 10 ** 3 * 8760 * 60 * 60  # meter/yrear
period = 248   #years.
def f(r):
    """
    The right hand side of our system of first order differential equation.
    returns a 1x4 array of information namely the xy positions and velocities.
    """
    x = r[0]
    s = r[1]
    y = r[2]
    p = r[3]
    fx = s
    fy = p
    fs = -G*M*(x/(np.sqrt(x**2+y**2))**3)
    fp = -G*M*(y/(np.sqrt(x**2+y**2))**3)
    
    return array([fx,fs,fy,fp],float)



tpoints = arange(0,period*1.01,H)  #time positions.
xpoints = []
ypoints = []
r = array([x_0,vx_0,y_0,vy_0],float)  #initial condition array.

# Do the "big steps" of size H
for t in tpoints:
    #recording the xy-positions before making new changes.
    xpoints.append(r[0])
    ypoints.append(r[2])

    # Do one modified midpoint step to get things started
    n = 1
    r1 = r + 0.5*H*f(r)
    r2 = r + H*f(r1)

    # The array R1 stores the first row of the
    # extrapolation table, which contains only the single
    # modified midpoint estimate of the solution at the
    # end of the interval
    R1 = empty([1,4],float)
    R1[0] = 0.5*(r1 + r2 + 0.5*H*f(r2))

    # Now increase n until the required accuracy is reached
    error = 2*H*delta
    while error>H*delta:

        n += 1
        h = H/n

        # Modified midpoint method
        r1 = r + 0.5*h*f(r)
        r2 = r + h*f(r1)
        for i in range(n-1):
            r1 += h*f(r2)
            r2 += h*f(r1)

        # Calculate extrapolation estimates.  Arrays R1 and R2
        # hold the two most recent lines of the table
        R2 = R1
        R1 = empty([n,4],float)
        R1[0] = 0.5*(r1 + r2 + 0.5*h*f(r2))
        for m in range(1,n):
            epsilon = (R1[m-1]-R2[m-1])/((n/(n-1))**(2*m)-1)
            R1[m] = R1[m-1] + epsilon
        error = abs(epsilon[0])

#     Set r equal to the most accurate estimate we have,
#     before moving on to the next big step
    r = R1[n-1]


#plotting the trajectory of Earth and related parameters:
plt.axis("equal")
plt.scatter((Au**-1)*np.array(xpoints),(Au**-1)*np.array(ypoints),label="Earth")
plt.plot((Au**-1)*np.array(xpoints),(Au**-1)*np.array(ypoints),color='r')
plt.scatter([0],[0],color="yellow",linewidth=10,label="The Sun")
plt.title("Orbit of Earth around the Sun",fontsize=14)
plt.xlabel("X (Au)",fontsize=14)
plt.ylabel("Y (Au)",fontsize=14)
print("difference between the max and min distance from the sun in the x direction is "+str(abs((max(xpoints)+min(xpoints))/Au))+" Au.")
plt.legend()
    
#plotting the trajectory of Pluto and related parameters:
plt.scatter((Au**-1)*np.array(xpoints),(Au**-1)*np.array(ypoints),label="Pluto",s=4)
plt.plot((Au**-1)*np.array(xpoints),(Au**-1)*np.array(ypoints),color='r',linewidth=0.5)
plt.scatter([0],[0],color="yellow",linewidth=10,label="The Sun")
plt.title("Orbit of Pluto around the Sun",fontsize=14)
plt.xlabel("X (Au)",fontsize=14)
plt.ylabel("Y (Au)",fontsize=14)
print("difference between the max and min distance from the sun in the x direction is "+str(abs((max(xpoints)+min(xpoints))/Au))+" Au.")
plt.legend()







    


