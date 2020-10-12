"""
Lab05 Question-2.
Author: Arya Kimiaghalam.
"""
import numpy as np
from numpy.fft import rfft2,irfft2
from pylab import imshow,gray,show
import matplotlib.pyplot as plt

initial_pic = np.loadtxt("blur.txt")

##Part-a:
#sigma = 25 in this case.

#Plotting the initial image (blury version).

plt.title("Blurry Version of the Picture",fontsize=14)
imshow(initial_pic)
gray()
show()

##Part-b:

"""
Comment: The picture we are analyzing is 1024x1024 in size so we must make the
size of this matrix 1024.
"""
grid = np.zeros((1024,1024),dtype=float)
def gaussian_spread(x,y,sigma):
    """
    This function is a Gaussian point spread function for varying values 
    of sigma.
    """
    return np.exp(-(x**2+y**2)/(2*sigma**2))
#filling up the grid with gaussian values:
for i in range(1024):
    for j in range(1024):
        grid[i-512,j-512] = gaussian_spread(i-512,j-512,25)   #sigma=25 here.

#visualizing the grid:
plt.title("Gaussian Spread Function",fontsize=14)
imshow(grid)
gray()
show()  #don't forget to give a title later!

##Part-c:
#Step-1: Fourier Transforming f(x,y) and the initial image:
unblurred_f_space = np.zeros((1024,513),dtype=complex)
grid_fft = rfft2(grid)
initial_pic_fft = rfft2(initial_pic)
#Step-2: making the fourier transformed matrix of the unblured function.
for i in range(1024):
    for j in range(513):
        if grid_fft[i,j] > 0.001:   #taking account for dividing by zero and small numbers. If ep<<1, then we don't touch the coefficient. 
            unblurred_f_space[i,j] = initial_pic_fft[i,j]/(grid_fft[i,j])
        else:
            unblurred_f_space[i,j] = initial_pic_fft[i,j]

#showing the unblured picture:
unblurred_r_space = irfft2(unblurred_f_space)
plt.title("Unblurred Version of the Picture",fontsize=14)
imshow(unblurred_r_space)
gray()
show()

