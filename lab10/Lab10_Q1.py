"""
Lab10
Question-1
Author: Arya Kimiaghalam
please run the function calls separately.
"""
import numpy as np
import matplotlib.pyplot as plt
import random
##########################Part-a##############################################:
               
def random_walk(pos_i,L):
    """
    random walk function for part-a.
    pos_i: initial position
    L: dimension of the confinement. 
    returns an updated position for the particle after each call.
    """
    #the coordinates of the current position:
    x = pos_i[0]
    y = pos_i[1]
    #generating a random number for the random walk:
    a = random.randint(1,4)
    if a == 1:
        #move right:
        x +=1
        if x == L:  #does not move if the particle crosses the right boundary.
            x = L-1
    if a == 2:
        #move up:
        y +=1
        if y == L:  #does not move if the particle crosses the upper boundary.
            y = L-1
    if a == 3:
        #move left:
        x -=1
        if x == -1:   #does not move if the particle crosses the left boundary.
            x = 0
    if a == 4:
        #move down:
        y -=1
        if y == -1:  #does not move if the particle crosses the bottom boundary.
            y = 0
    return [x,y]


def single_traj_plotter(start,L,N):
    """
    returns the trace of where a randomly walking particle has traveled over
    N steps on an LxL grid.
    start: starting point of the particle.
    L: dimension of the grid.
    N: iterations.
    """
    inter_pos = start  #intermediate position subject to update.
    spots = np.zeros((L,L))  #trace track matrix, subject to update.
    for i in range(N):
        x = inter_pos[0]
        y = inter_pos[1]
        spots[x,y] = 1  # "lighting up" locations that the particle was at.
        new_pos = random_walk(inter_pos,L)  
        inter_pos = new_pos  #updating intermediate position.
    #graphing the track matrix:
    plt.imshow(spots,cmap="Reds")
    plt.title("Particle Track")
    plt.xlabel("x")
    plt.ylabel("y")

#generating results for part-a:
single_traj_plotter([50,50],101,5000)
####################################Part-b&c##################################:
def bp(point,L):
    """
    returns True if the point is on the boundary and False if not.
    L: dimension of grid.
    """
    x = point[0]
    y = point[1]
    values = [0,L-1]  #possible boundary coordinate values.
    if x in values or y in values:
        return True
    else:
        return False

def surround(pos):
    """
    returns a set of the adjacent points of a particle.
    """
    lst = []
    x = pos[0]
    y = pos[1]
    lst.append([x+1,y])  #right.
    lst.append([x-1,y])  #left.
    lst.append([x,y+1])  #up.
    lst.append([x,y-1])  #down.
    return lst

def common(lst1, lst2):
    """
    Returns True if the two lists have an element in common, returns False otherwise.
    """
    r = any([x in lst1 for x in lst2])
    return r
    
"""
    Pseudocode for DLA:
    1) Receive the dimension of the grid and starting point of all the particles.
    
    2) create a list to store the locations of the stuck particles.
    
    3) while the starting point is not filled (not in the stuck_par_list)
      introduce a new point each time.
    
    4) while the location of the point is not a boundary point or there is no
    particle on the right, left, above and below it, update the location using
    the random walk function.
    
    5) if the newly updated location is a boundary point, add it to the stuck_par_list
    and introduce another particle.
    
    6) if there is at least one stuck particle near the newly updated position
    of our particle, record the position the particle is at by appending it to
    the  stuck_par_list.
    
    7) if those conditions are not met, continue updating the location of the 
    same particle another time and do not introduce a new particle.
    
    8) continute this process until the starting position is in the stuck_par_list.
    
    9) return the location of the stuck particles at the end.
"""



def plotter(coords,L):
    """
    plots the final position of all stuck particles.
    """
    image = np.zeros((L,L))   #initializing the image array.
    for coord in coords:
        image[coord[0],coord[1]] = 1  #marking the positions.
    plt.imshow(image)
    plt.plot("Particle Map")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    
    
    
def movie_plotter(tracks,L):
    """
    Generates a movie-like sequence of the moving particles until all of them
    are stuck. We use this for the animation in part-b and get our tracks from
    DLA_b function.
    tracks: a list of the tracks of the particles over time.
    """
    background = np.zeros((L,L))  #background for every new particle that starts from the center. subject to update.
    track_ends = []  #final location of every track.
    for track in tracks:
        track_ends.append(track[-1])
    
    
    for i in range(len(tracks)-1):
        x_b = track_ends[i][0]
        y_b = track_ends[i][1]
        background[x_b,y_b] = 1  #updating the background for newly introduced particles.
        for pos in tracks[i+1]:
            z = np.zeros((L,L))  #momentary image (snapshot).
            x = pos[0]
            y = pos[1]
            z[x,y] = 1
            plt.imshow(z+background)
            plt.show()
            
            

    
            
############################################################################################################################ 
def DLA(L,start):
    """
    returns a list of the locations of the stuck particles.
    L: dimension.
    start: starting point.
    """
    stuck_particles = []  #list of the location of stuck particles.
    while start not in stuck_particles:
        locations =[]     #location of one of the introduced particles.
        locations.append(start)  #first point of it is always the starting point.
        inter_pos = start  #intermediate position starts from the starting point.
        stuck = False  #halting criterion.
        while stuck == False:
            new_pos = random_walk(inter_pos,L)
            inter_pos = new_pos  #updating position.
            locations.append(inter_pos)  #recording updated position.
            if common(surround(locations[-1]),stuck_particles) == True:  #if there happens to be an adjacent stuck particle.
                stuck_particles.append(locations[-1]) #we record where we were to the stuck particle list and end second loop.
                stuck = True
            elif bp(locations[-1],L) == True:  #if we reach the boundary:
                stuck_particles.append(locations[-1])  #we record where we end up.
                stuck = True
            else:
                pass


    return stuck_particles


def DLA_b(L,start):
    """
    Same as the DLA function except that it returns all the trajectories
    not just the location of the stuck particles. this is used for generating
    movies for part b to feed the movie_plotter function.
    """
    stuck_particles = []  #list of the location of stuck particles.
    all_tracks = []
    while start not in stuck_particles:
        locations =[]     #location of one of the introduced particles.
        locations.append(start)  #first point of it is always the starting point.
        inter_pos = start  #intermediate position starts from the starting point.
        stuck = False  #halting criterion.
        while stuck == False:
            new_pos = random_walk(inter_pos,L)
            inter_pos = new_pos  #updating position.
            locations.append(inter_pos)  #recording updated position.
            if common(surround(locations[-1]),stuck_particles) == True:  #if there happens to be an adjacent stuck particle.
                stuck_particles.append(locations[-1]) #we record where we were to the stuck particle list and end second loop.
                all_tracks.append(locations)  #recording track.
                stuck = True
            elif bp(locations[-1],L) == True:  #if we reach the boundary:
                stuck_particles.append(locations[-1])  #we record where we end up.
                all_tracks.append(locations)  #recording track.
                stuck = True
            else:
                pass


    return all_tracks






#generating result for part-b:
lst = DLA(101,[50,50])
plotter(lst,101)

#to get the first 100 stuck particle:
plotter(lst[:100],101)


#generating results for part-c (similarly for L= 151):
lst = DLA(201,[100,100])
plotter(lst,201)

