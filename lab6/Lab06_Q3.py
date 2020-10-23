#Question 3
#Author: Andrey Remorov

#in this question, I use an adjusted version of the code used in question 2 to
#analyze the behaviour of a many molecule system.

#Part b
#Description: This code gets the total energy at each timestep, and stores a 
#plot of it in the same folder as this python file.


import numpy as np
import matplotlib.pyplot as plt


#generalized function of F() used in quesition 2:
def F_g(pos_list): #2nx1 list, for n particles (each has an x and y component)
    """
    second time derivative of the position vector.
    """
    a = np.zeros(len(pos_list)) #acceleration list
    
    for i in range(len(a)//2):
        for j in range(i+1, len(a)//2): #make sure j>i to prevent overcounting
            pos_1 = [pos_list[2*i], pos_list[2*i+1]]
            pos_2 = [pos_list[2*j], pos_list[2*j+1]]
            x = pos_2[0] - pos_1[0]
            y = pos_2[1] - pos_1[1]
            c1 = (48*x*((x**2+y**2)**(-7))) - (24*x*((x**2+y**2)**(-4)))#dF/dx
            c2 = (48*y*((x**2+y**2)**(-7))) - (24*y*((x**2+y**2)**(-4)))#dF/dy
            a[2*i] += -c1 #a_x1
            a[2*i+1] += -c2 #a_y1
            a[2*j] += c1 #a_x2
            a[2*j+1] += c2 #a_y2
    return a 

def PE(p):#gets potential energy of the system, given the position list p
    pe = 0
    for i in range(len(p)//2):
        for j in range(i+1, len(p)//2): #make sure j>i to prevent overcounting
            r = np.sqrt((p[2*j]-p[2*i])**2 + (p[2*j+1]-p[2*i+1])**2) #displacement between pair 
            pe+=4*((1/r)**12 - (1/r)**6)
    return pe

def KE(v):#get kinetic energy of system, given the velocity list v 
    #(assume m=1 for all particles):
    ke = 0
    for i in range(len(v)):
        ke+= (v[i]**2)/2 #m=1
    return ke

#use modified xy_coord() function that includes storing energy values at each 
#timestep:
def xy_coord_3(pos_list):
    """
    returns trajectory of each particle
    """
    h = 0.01
    r = pos_list
    r_list = [] #trajectory list
    v = 0.5*h*F_g(r) #v(t+h/2), t=0 here
    E_list = [PE(r)] #energy list, with initial energy (just potential at t=0)
    for i in range(1000):
        r_list.append(list(r))
        r+=h*v #get r(t+h)
        k = h*F_g(r)
        v_mid = v+k/2 #get v(t+h)
        v+=k #get v(t+3h/2)
        E = PE(r) + KE(v_mid)
        E_list.append(E)
    return [np.array(r_list), E_list]


#generate the square grid using code from lab instructions:
N = 16
Lx = 4.0
Ly = 4.0
dx = Lx/np.sqrt(N)
dy = Ly/np.sqrt(N)
x_grid = np.arange(dx/2, Lx, dx)
y_grid =  np.arange(dy/2, Ly, dy)
xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
x_initial = xx_grid.flatten()
y_initial = yy_grid.flatten()

#get initial position vector:
r_initial = np.zeros(len(x_initial)*2)
for i in range(len(x_initial)):
    r_initial[2*i] = x_initial[i]
    r_initial[2*i+1] = y_initial[i]
    
traj = xy_coord_3(r_initial)

#plot energy wrt timestep:
plt.plot(traj[1]) #wthin 1%, nice
plt.title("Total Energy")
plt.xlabel("Timestep")
plt.ylabel("Energy (dimensionless)")
plt.savefig("Q3b_E.pdf")
plt.clf()



#Part c
#Description: This code enforces perioduc boundary conditions, and outputs the
#plot of trajectories of the particles for the first 1000 timesteps.


def F_gp(pos_list,Lx,Ly): 
    #getting forces for periodic boundary conditions [0,Lx]x[0,Ly]

    #make difference x, y arrays to easily change the x and y coordinates by Lx,Ly
    Dx, Dy = np.zeros(len(pos_list)), np.zeros(len(pos_list))
    Dx[::2] = Lx
    Dy[1::2] = Ly
    
    #make the periodic boundary position list:
    per_pos = pos_list #real particles
    #image particles:
    per_pos = np.append(per_pos, pos_list-Dx-Dy) #bottom left
    per_pos = np.append(per_pos, pos_list-Dx) #centre left
    per_pos = np.append(per_pos, pos_list-Dx+Dy) #top left
    per_pos = np.append(per_pos, pos_list-Dy) #bottom centre
    per_pos = np.append(per_pos, pos_list+Dy) #top centre
    per_pos = np.append(per_pos, pos_list+Dx-Dy) #bottom right
    per_pos = np.append(per_pos, pos_list+Dx) #centre right
    per_pos = np.append(per_pos, pos_list+Dx+Dy) #top right
    
    
    a = np.zeros(len(per_pos)) #acceleration list
    
    for i in range(len(a)//2):
        for j in range(i+1, len(a)//2): #make sure j>i to prevent overcounting
            pos_1 = [per_pos[2*i], per_pos[2*i+1]]
            pos_2 = [per_pos[2*j], per_pos[2*j+1]]
            x = pos_2[0] - pos_1[0]
            y = pos_2[1] - pos_1[1]
            c1 = (48*x*((x**2+y**2)**(-7))) - (24*x*((x**2+y**2)**(-4)))
            c2 = (48*y*((x**2+y**2)**(-7))) - (24*y*((x**2+y**2)**(-4)))
            a[2*i] += -c1
            a[2*i+1] += -c2
            a[2*j] += c1
            a[2*j+1] += c2
    return a[:len(pos_list)]#return acceleration for real particles, not images

def xy_coord_4(pos_list,Lx,Ly): #adjusted for periodic boundary conditions
    h = 0.01
    r = pos_list
    r_list = [] #trajectory list
    v = 0.5*h*F_gp(r,Lx,Ly) #v(t+h/2), t=0 here
    E_list = [PE(r)] #energy list, with initial energy (just potential at t=0)
    for i in range(1000):
        r_list.append(list(r))
        r+=h*v #get r(t+h)
        k = h*F_gp(r,Lx,Ly)
        v_mid = v+k/2 #get v(t+h)
        v+=k #get v(t+3h/2)
        E = PE(r) + KE(v_mid)
        E_list.append(E)
        r[::2]=np.mod(r[::2], Lx) #bring x values back into [0,Lx] domain
        r[1::2]=np.mod(r[1::2], Ly) #bring y values back into [0,Ly] domain
    return [np.array(r_list), E_list]


#get initial position vector:
r_initial = np.zeros(len(x_initial)*2)
for i in range(len(x_initial)):
    r_initial[2*i] = x_initial[i]
    r_initial[2*i+1] = y_initial[i]
    
traj = xy_coord_4(r_initial,Lx,Ly)

#plot trajectories:
for i in range(len(x_initial)):
    plt.plot(traj[0][:,2*i],traj[0][:,2*i+1], ".", markersize=.5)
    plt.axis("equal")
plt.title("Trajectories in periodic boundary conditions")
plt.xlabel(r"[0,$L_x$] Domain")
plt.ylabel(r"[0,$L_y$] Domain")
plt.savefig("Q3c_traj.pdf")
plt.clf()


