"""
Lab06
Question-2
Author: Arya Kimiaghalam
NOTE: Please run the plotting codes of set1,set2 and set3 separately.
"""
import numpy as np
import matplotlib.pyplot as plt
#Part-a:
"""
Nothing to submit.
"""

#Part-b:
"""
##Pseudocode:
1- Write a function for the second time derivative of the x1,y1,x2 and y2 coordiates 
   ,based on the Leonard_Jones Equation, that returns [x2'',y2'',x1'',y1'']
   (i.e. the right side of our differential equation).
2- Write a function that does the following:
   a) Recieve initial positions for both particles from keyboard.
   b) Makes the initial condition to a single 1x4 array
      of the coordinates of both particle 1 and 2, and set if to an updatable
      variable r.
   c) set the initial first time derivative of x1,y1,x2 and y2 to zero (i.e. v).
   d) update v (i.e. first time derivative of the separation distance) and 
       itterate using the Velret algorithm until done.
   e) return all x and y coordinates (i.e. trajectories of both particles).
      END.
"""


def F(pos_1,pos_2):
    """
    second time derivative of the position vector (Right side of the 
    differential equation).
    """
    x = pos_2[0] - pos_1[0]
    y = pos_2[1] - pos_1[1]
    c1 = (48*x*((x**2+y**2)**(-7))) - (24*x*((x**2+y**2)**(-4)))
    c2 = (48*y*((x**2+y**2)**(-7))) - (24*y*((x**2+y**2)**(-4)))
    return np.array([c1,c2,-1*c1,-1*c2])   # second time derivative of the vector: [x2,y2,x1,y1].

def xy_coord(pos_i_1,pos_i_2):
    """
    returns x1,y1,x2,y2 of both particles.
    """
    h = 0.01
    r = np.array([pos_i_2[0],pos_i_2[1],pos_i_1[0],pos_i_1[1]])
    r_lst = []
    v = [np.array([0,0,0,0])]  #gets updated through the loop.
    v.append(np.array(v[-1])+0.5*h*F(pos_i_1,pos_i_2))
    for i in range(100):
        r_lst.append(r)
        r = r + h*np.array(v[-1])
        k = h*F([r[2],r[3]],[r[0],r[1]])
        v.append(np.array(v[-1])+k)
    x2 = []
    y2 = []
    x1 = []
    y1 = []
    for item in r_lst:
        x2.append(item[0])
        y2.append(item[1])
        x1.append(item[2])
        y1.append(item[3])
    return [np.array(x1),np.array(y1),np.array(x2),np.array(y2)]


#plotting the trajectories separately:
#Initial conditions:
set1 = [[4,4],[5.2,4]]
set2 = [[4.5,4],[5.2,4]]
set3 = [[2,3],[3.5,4.4]]
#set1:
plt.plot(xy_coord(set1[0],set1[1])[2],xy_coord(set1[0],set1[1])[3],'.',markersize=5,c='r',label="Particle-2")
plt.title("Trajectory of Particle-1,2 Under Condition i",fontsize=14)
plt.xlabel("X",fontsize=14)
plt.ylabel("Y",fontsize=14)
plt.plot(xy_coord(set1[0],set1[1])[0],xy_coord(set1[0],set1[1])[1],'.',markersize=5,label="Particle-1")
plt.legend()
plt.show()
plt.plot(np.arange(0,1,0.01),xy_coord(set1[0],set1[1])[0],'.')
plt.title("X-coordinate of Particle-1 under condition i",fontsize=14)
plt.xlabel("Time (s)",fontsize=14)
plt.ylabel("X",fontsize=14)

#set2:
plt.plot(xy_coord(set2[0],set2[1])[2],xy_coord(set2[0],set2[1])[3],'.',markersize=5,c='r',label="Particle-2")
plt.title("Trajectory of Particle-1,2 Under Condition ii",fontsize=14)
plt.xlabel("X",fontsize=14)
plt.ylabel("Y",fontsize=14)
plt.plot(xy_coord(set2[0],set2[1])[0],xy_coord(set2[0],set2[1])[1],'.',markersize=5,label="Particle-1")
plt.legend()
plt.show()
plt.plot(np.arange(0,1,0.01),xy_coord(set2[0],set2[1])[0],'.')
plt.title("X-coordinate of Particle-1 under condition ii",fontsize=14)
plt.xlabel("Time (s)",fontsize=14)
plt.ylabel("X",fontsize=14)

#set3:
plt.plot(xy_coord(set3[0],set3[1])[2],xy_coord(set3[0],set3[1])[3],'.',markersize=5,c='r',label="Particle-2")
plt.title("Trajectory of Particle-1,2 Under Condition iii",fontsize=14)
plt.xlabel("X",fontsize=14)
plt.ylabel("Y",fontsize=14)
plt.plot(xy_coord(set3[0],set3[1])[0],xy_coord(set3[0],set3[1])[1],'.',markersize=5,label="Particle-1")
plt.legend()
plt.show()
plt.plot(np.arange(0,1,0.01),xy_coord(set3[0],set3[1])[0],'.')
plt.title("X-coordinate of Particle-1 under condition iii",fontsize=14)
plt.xlabel("Time (s)",fontsize=14)
plt.ylabel("X",fontsize=14)






