"""
Author: Arya Kimiaghalam
Code for Question 3:
Please run the plotting codes separately.
"""
import numpy as np
import matplotlib.pyplot as plt
from pylab import imshow,gray,show
##Question-3-a:
#The separation distance between the slits is (pi/alpha)

##Question-3-b:
def q_1(u):
    """
    Returns the transmission function q(u) = sin^2(alpha*u) at position u for a grating
    that its slits have a separation of 20 micro meters.
    input u should be in meters.
    """
    return (np.sin(((np.pi/(20*10**-6))*u)))**2

#Question-3-c:

"""
Pseudocode:
    1-grating-space vertical position value is recorded from keyboard.
    2-vertical position on the screen is recorded from keyboard.
    3- wavelength, focal point distance and the relavant transmission function is recorded from keyboard.
    4- calculate the value of the inner function at a variety of vertical positions of the grating, namely -w/2 to +w/2.
    5- Use those values to calculate the integral numerically through Simpson's method, to find the intensity of light at x above the mid line on the screen.
    6- Find the square modulus of the integral value.
    7- Repeat for different values of vertical position on the projection screen and compile it as an array.
"""
def inner_function(u,x,lam,f,q):
    """
    Rreturns the value of the function that needs ti be integrated in order to find I(x)
    at the end.
    u: distance from the central axis on the difraction grid; an argument for q(u).
    x: distance from the central axis on the screen; an argument for I(x); (fixed in the integration).
    lam: wavelength in meters.
    f: focal distance of lens (in meters).
    q: The function q can alter between q_1(u), q_2(u) and q_3(u).
    """
    return (np.sqrt(q(u)))*np.exp((1j*2*np.pi*x*u)/(lam*f))  #The function q can alter between q(u) and q_2(u).

wavelength = 500*(10**-9)
slit_count = 10
focal = 1
screen_width = 0.1  
grating_width = slit_count*20*(10**-6) #assuming slits are very small compared to separations and their share of the total length is almost none.

def integral_calc(w,x,lam,f,N,q):
    """
    calculates the integral of the inner_function.
    w: diffraction grating width.
    N: number of x_points on the x_axis (for the numerical calculation).
    Other variables are similar to the variables of the inner function.
    """
    h = w/N
    #we use Simpson's rule for integration here:
    odd = []
    even = []
    for k in range(1,N,2):
        odd.append(4*inner_function((-w/2)+k*h,x,lam,f,q))
    for k in range(2,N,2):
        even.append(2*inner_function((-w/2)+k*h,x,lam,f,q))
    return (1/3)*h*(inner_function(-w/2,x,lam,f,q)+inner_function(w/2,x,lam,f,q)+sum(odd)+sum(even))
def I(w,x,lam,f,N,q):
    """
    intensity function for height x (in meters)
    Calculates I(x) based on inner function's parameters.
    """
    return abs(integral_calc(w,x,lam,f,N,q))**2

def Spectrum_data(w,d,lam,f,N,q):
    """
    Calculates the intensity over a range of x-values.
    d: screen width in meters.
    """
    Intensities = []
    x_pos = np.arange(-d/2,d/2,0.00001) #increments of 10^-5 meters.
    for point in x_pos:
       Intensities.append(I(w,point,lam,f,N,q))
    return np.array(Intensities)
##Q-3-d (incomplete):

diff_data = Spectrum_data(grating_width,0.1,500*10**-9,1,100,q_1)  #generates one dimensional diffraction data.

data = np.empty((5555,10000),float)   #used to stack up linear (one dimensional) diffraction data to creat a 2D image of patterns.
for i in range(5555):     #stacking up!
    for j in range(10000):
        data[i,j] = diff_data[j]
imshow(data, extent = [0,10,0,5])
gray()
plt.colorbar()
plt.title("Interference Pattern for q_1(u)")
plt.xlabel("Vertical position from the top of the screen (cm)")
show()
##Q-3-e-i:
def q_2(u):
    """
    Returns the transmission function q(u) = sin^2(alpha*u) at position u for a grating
    that its slits have a separation of 20 micro meters.
    input u should be in meters.
    """

    return (np.sin(((np.pi/(20*10**-6))*u))*np.sin((10*10**-6)*u))**2

diff_data_2 = Spectrum_data(grating_width,0.1,500*10**-9,1,100,q_2)   #generating second set of data, based on q_2.

data = np.empty((5555,10000),float)   ##used to stack up linear (one dimensional) diffraction data to creat a 2D image of patterns.
for i in range(5555):    #stacking up!
    for j in range(10000):
        data[i,j] = diff_data_2[j]
imshow(data, extent = [0,10,0,5])
gray()
plt.colorbar()
plt.title("Interference Pattern for q_2(u)")
plt.xlabel("Vertical position from the top of the screen (cm)")
show()
##Q-3-e-ii (incomplete):
def q_3(u):
    """
    Returns the transimission function for two
    separated single slits of size 10 and 20 
    micrometers with a 60nm separation distance.
    returns 1 at slit locations (100% transmission) and 0 at the walls (no transmission).
    """
    if u>=40*(10**-6):
        return float(0.0)
    elif 40*(10**-6)>u>=30*(10**-6):
        return float(1.0)
    elif 30*(10**-6) > u > -30*(10**-6):
        return float(0.0)
    elif -30*(10**-6)>=u>-50*(10**-6):
        return float(1.0)
    elif -50*(10**-6)>=u:
        return float(0.0)
    
diff_data_3 = Spectrum_data(grating_width,0.1,500*10**-9,1,100,q_3)  #generating third set of diffraction data, based on q_3.

#Stacking up the 1-d array into a thick intensity pattern. 
data = np.empty((5555,10000),float)  ##used to stack up linear (one dimensional) diffraction data to creat a 2D image of patterns.
for i in range(5555):
    for j in range(10000):
        data[i,j] = diff_data_3[j]
imshow(data, extent = [0,10,0,5])
gray()
plt.colorbar()
plt.title("Interference Pattern for q_3(u)")
plt.xlabel("Vertical position from the top of the screen (cm)")
show()

