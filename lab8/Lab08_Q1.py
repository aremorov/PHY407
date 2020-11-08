"""
Lab08 Q1
Author: Arya Kimiaghalam
Please run the plotting codes separately.
"""
import numpy as np
import matplotlib.pyplot as plt
# Relevant constants:
M = 100         # Grid numbers
V = 1.0         # magnitude of the voltage at the nodes
target = 10**(-6)   # Target accuracy

# Create array of potentials:
phi = np.zeros([M+1,M+1],float)

#For no overrelaxation set w = 0. I avoided writing two set of codes for the sake of reducing repetition.
delta = 10.0  #random number for delta, should be bigger than the target to start the while loop going.
w = 0.9   #value of omega subject to change. Set w = 0 for no overrelaxation.
while delta>target:

    delta = 0.0
    # Calculate new values of the potential one by one
    for i in range(1,M):
        for j in range(1,M):
            
            if j==(M/5) and i>(M/5) and i<(4*M/5):   #making sure we have the right values for the positive node.
                phi[i,j] = V
            elif j==(4*M/5) and i>(M/5) and i<(4*M/5):     #making sure we have the right values for the negative node.
			          phi[i,j] = -V
            else:
	            err = (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])/4 - phi[i,j]   #the error of this itteration of the Gauss-Seidel (un-overrelaxed) method.
	            phi[i,j] = phi[i,j] + (1+w)*err  #now we adapt the overrelaxation method with a pre-set value of omega (i.e. w).
            if err>delta:   # making sure that the error is always greater than delta while the loop is running.
                delta = err   #updating error




#Finding the field lines
##Making a XY meshgrid and associate the grid points with electric field vectors.
x = np.linspace(-5, 5, 101) # does not include zero
y = np.linspace(-5, 5, 101)
X, Y = np.meshgrid(x, y)

#calculating the x and y electric components:
Ey, Ex = np.gradient(-phi, y, x) # careful about order
#plotting the electric vector field and the ontour plot of the potentials:
fig = plt.figure(figsize=(7, 6))
strm = plt.streamplot(X, Y, Ex, Ey, color = phi, linewidth=2, cmap='hot')
cbar = fig.colorbar(strm.lines)
cbar.set_label('Potential $V$',fontsize=14)
plt.title('Electric field lines',fontsize=14)
plt.xlabel('$x$'+" (cm)",fontsize=14)
plt.ylabel('$y$'+" (cm)",fontsize=14)
plt.axis('equal')
plt.tight_layout()

#the voltage levels we want to plot:
heights = np.arange(-1,1,0.1)


#plotting the contour plot
fig = plt.figure(figsize=(7, 6))
c = plt.contour(X,Y,phi,levels=heights,cmap='hot')
cc = fig.colorbar(c)
cc.set_label('Potential $V$',fontsize=14)
plt.title('Voltage Contour Plot',fontsize=14)
plt.xlabel('$x$'+" (cm)",fontsize=14)
plt.ylabel('$y$'+" (cm)",fontsize=14)
plt.axis('equal')
plt.tight_layout()

plt.show()

