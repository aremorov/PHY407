"""
Lab10
Question-2
author: Arya Kimiaghalam
"""
import numpy as np
import random

def sphere_volume(d):
    """
    calculates the volume of an n-dimensional unit sphere
    d: dimension of interest.
    """
    points = np.zeros((10**6,d))  #records the generated points.
    count = []   #keep count of the number of particles that hit inside the volume.
    for i in range(10**6):   #looping over a million points.
        for j in range(d):  #generating position vectors (d-components needed).
            points[i,j] = random.random()
    
    points2 = points**2  #finding the square of the coordinates.
    for i in range(10**6):
        if sum(points2[i]) <= 1:  #finding the modulus square
            count.append(1)   #marking as "hit"
        else:
            count.append(0)  #marking as "miss"
    
    var_f = np.var(np.array(count))  #finding the variance of the values of our volume function over the iterations.
    error = (2**d)*np.sqrt(var_f)/(1000)  #finding the value of the error for the mean value Monte Carlo method.
    
    return [(sum(count)*(2**d))/(10**6),error]


#generating result:
print("volume is "+str(sphere_volume(10)[0])+" error is "+str(sphere_volume(10)[1])+" for a 10 dimensional unit sphere.")





