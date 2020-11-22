"""
Lab09 
Question-2
Author: Arya Kimiaghalam
Notes:
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft,irfft, fft2

###PART-B,C###:

tao = 0.01  #subject to change.

def Jz(m,n,Lx,Ly,J0,w,t,P):
    """
    returns the flux for the (m,n) mode as a 2D array.
    """
    J = np.zeros((P,P),dtype=float)  #place holder for calculated regional flux values.
    #looping over cells in the 2D cavity:
    for i in range(1,P-1):
        for j in range(1,P-1):
            x = i*(Lx/P)
            y = j*(Ly/P)
            J[i,j] = J0*(np.sin((np.pi)*m*x/Lx)*np.sin((np.pi)*n*y/Ly)*np.sin(w*t))
    return J
#Initial values for the cavity and also other variables.
P = 32  #number if intervals for each side.
Lx = 1  #x-length
Ly = 1  #y-length
J0 = 1  #J0 value
m = 1  #m-mode
n = 1  #n-mode
c = 1  #velocity if light
w = 3.75  #omega value. Subject to change to the normal frequency for other calculations.
def fc_update(E,X,Y,J):
    """
    Updates Fourier coefficients of everything except flux.
    E,X,Y represent the previous index 2D matrices and J is the 2D flux array
    for that particular time.
    """
    E_new = np.zeros((len(E),len(E)),dtype=complex)   #New 2D array of E fourier coefficients.
    X_new = np.zeros((len(E),len(E)),dtype=complex)   #New 2D array of Hx fourier coefficients.
    Y_new = np.zeros((len(E),len(E)),dtype=complex)   #New 2D array of Hy fourier coefficients.
    Dx = (np.pi)*c*tao/(2*Lx)  #constant Dx
    Dy = (np.pi)*c*tao/(2*Ly)  #constant Dy
    #looping over the grid:
    for p in range(len(E)):
        for q in range(len(E)):
            k1 = ((((1-(p**2)*(Dx**2)-(q**2)*(Dy**2))*E[p,q]) + 2*q*Dy*X[p,q] + 2*p*Dx*Y[p,q] + tao*J[p,q])/(1+(p**2)*(Dx**2)+(q**2)*(Dy**2)))
            k2 = X[p,q] - q*Dy*(k1+E[p,q])
            k3 = Y[p,q] - p*Dx*(k1+E[p,q])
            #producing the new coefficients for the [p,q] position.
            E_new[p,q] = k1
            X_new[p,q] = k2
            Y_new[p,q] = k3
    return [E_new,X_new,Y_new]

###############################################################################
#The following functions are borrowed from Mark Newman's code posted on Quercus:
def dct(y):   
    """
    Direct Cosine Transform of 1D array.
    """
    N = len(y)    
    y2 = np.empty(2*N,float)    
    y2[:N] = y[:]    
    y2[N:] = y[::-1]    
    c = rfft(y2)    
    phi = np.exp(-1j*np.pi*np.arange(N)/(2*N))    
    return np.real(phi*c[:N])

def idct(a):  
    """
    Inverse Direct Cosine Transform of 1D array.
    """
    N = len(a)    
    c = np.empty(N+1,complex)    
    phi = np.exp(1j*np.pi*np.arange(N)/(2*N))    
    c[:N] = phi*a    
    c[N] = 0.0    
    return irfft(c)[:N]



def dst(y):    
    """
    Direct sine Transform of 1D array.
    """
    N = len(y)    
    y2 = np.empty(2*N,float)    
    y2[0] = y2[N] = 0.0    
    y2[1:N] = y[1:]    
    y2[:N:-1] = -y[1:]    
    a = -np.imag(rfft(y2))[:N]    
    a[0] = 0.0    
    return a

def idst(a):  
    """
    Inverse Direct sine Transform of 1D array.
    """
    N = len(a)    
    c = np.empty(N+1,complex)    
    c[0] = c[N] = 0.0    
    c[1:N] = -1j*a[1:]    
    y = irfft(c)[:N]    
    y[0] = 0.0    
    return y


###############################################################################
def sinsin(A):
    """
    DST tranformation along the rows and columns for a 2D array.
    """
    l = len(A[0])
    sin_1 = np.zeros((l,l))
    sin_2 = np.zeros((l,l))
    #first row transforms along the rows:
    for i in range(l):
        sin_1[i] = dst(A[i])
    #Second transforms along the columns:
    for j in range(l):
        sin_2[:,j] = dst(sin_1[:,j])
    return sin_2

def sincos(A):
    """
    DST tranform along the rows and DCT along the columns.
    """
    l = len(A[0])
    sin_1 = np.zeros((l,l))
    cos_2 = np.zeros((l,l))
    #first row sin transform along the rows:
    for i in range(l):
        sin_1[i] = dst(A[i])
    #second cos transform for columns:
    for j in range(l):
        cos_2[:,j] = dct(sin_1[:,j])
    return cos_2

def cossin(A):
    """
    DCT transform along the rows and DST along the columns.
    """
    l = len(A[0])
    cos_1 = np.zeros((l,l))
    sin_2 = np.zeros((l,l))
    #first row sin transform along the rows:
    for i in range(l):
        cos_1[i] = dct(A[i])
    #second cos transform for columns:
    for j in range(l):
        sin_2[:,j] = dst(cos_1[:,j])
    return sin_2

def i_sinsin(B):
    """
    Inverse function to sinsin.
    """
    l = len(B[0])
    sin_1 = np.zeros((l,l))
    sin_2 = np.zeros((l,l))
    #first inverse transform along the columns:
    for i in range(l):
        sin_2[:,i] = idst(B[:,i])
    #second inverse transform along the columns.
    for j in range(l):
        sin_1[j] = idst(sin_2[j])
    return sin_1

def i_sincos(B):
    """
    Inverse function to sincos.
    """
    l = len(B[0])
    cos_2 = np.zeros((l,l))
    sin_1 = np.zeros((l,l))
    #first inverse transform of columns:
    for i in range(l):
        cos_2[:,i] = idct(B[:,i])
    #second inverse transformation of rows: 
    for j in range(l):
        sin_1[j] = idst(cos_2[j])
    return sin_1


def i_cossin(B):
    """
    Inverse function to cossin.
    """
    l = len(B[0])
    sin_2 = np.zeros((l,l))
    cos_1 = np.zeros((l,l))
    #first inverse transformation of columns:
    for i in range(l):
        sin_2[:,i] = idst(B[:,i])
    #second inverse transformation of rows:
    for j in range(l):
        cos_1[j] = idct(sin_2[j])
    return cos_1

def EH_update(w,t):
    """
    updating E, Hx and Hy for time t.
    """
    N = t/tao  #number of time intervals for our input t.
    num = 0   #temporary time subject to update.
    E_inter = np.zeros((P,P))  #place holder for electric fields subject to update.
    Hx_inter = np.zeros((P,P)) #place holder for Hx subject to update.
    Hy_inter = np.zeros((P,P)) #place holder for Hy subject to update.
    while num < N:
        J = Jz(m,n,Lx,Ly,J0,w,num*tao,P)  #the flux matrix for an intermediate time t = num.
        J_hats = sinsin(J)  #fourier (double sin) transform coefficients of the 2D flux array.
        E_hat_inter = sinsin(E_inter) #updating E.
        Hx_hat_inter = sincos(Hx_inter) #updating Hx.
        Hy_hat_inter = cossin(Hy_inter) #updating Hy.
        #updating the fourier coefficients for the next step of the loop:
        E_hat_upd = fc_update(E_hat_inter,Hx_hat_inter,Hy_hat_inter,J_hats)[0]
        Hx_hat_upd = fc_update(E_hat_inter,Hx_hat_inter,Hy_hat_inter,J_hats)[1]
        Hy_hat_upd = fc_update(E_hat_inter,Hx_hat_inter,Hy_hat_inter,J_hats)[2]
        #getting the actual values of the updated values (array) of E,Hx,Hy:
        E_inter = i_sinsin(E_hat_upd)
        Hx_inter = i_sincos(Hx_hat_upd)
        Hy_inter = i_cossin(Hy_hat_upd)
        num = num + 1  #updating time by one time step.
    return [E_inter,Hx_inter,Hy_inter]


#getting the values of fields at particular locations over time:   
Hx = []
Hy = []
Ez = []
times = np.arange(0,20,10*tao)  #range of time used.
for i in times:
    Hx.append(EH_update(w,i)[1][31,15])  #value of Hx at the bottom-middle boundary.
    Hy.append(EH_update(w,i)[2][15,0])   #value of Hy at the left-middle boundary.
    Ez.append(EH_update(w,i)[0][15,15])  #value of Ez at the center of the cavity.

#plotting the fields over time:
plt.plot(times,Hx)
plt.title("Hx vs. Time",fontsize=14)
plt.xlabel("Time (s)",fontsize=14)
plt.ylabel("Hx",fontsize=14)
plt.show()

plt.plot(times,Hy)
plt.title("Hy vs. Time",fontsize=14)
plt.xlabel("Time (s)",fontsize=14)
plt.ylabel("Hy",fontsize=14)
plt.show()

plt.plot(times,Ez)
plt.title("Ez vs. Time",fontsize=14)
plt.xlabel("Time (s)",fontsize=14)
plt.ylabel("Ez",fontsize=14)
plt.show()

###Part-d###:
#plotting  the maximum amplitude of the electric field for varying omega:

omegas = np.arange(0,9,0.25)  #omega values.
max_amp = []

for omega in omegas:
  Ez = []
  for time in np.arange(0,10,0.5):
    Ez.append(EH_update(omega,time)[0][15,15])
  max_amp.append(max(Ez))

###Part-e###:
#plotting the amplitude values in respect to omega:
plt.plot(omegas,max_amp)
plt.title("Maximum Amplitude of the Electric Field vs. Omega",fontsize=12)
plt.xlabel("w",fontsize=14)
plt.ylabel("Max Amplitude",fontsize=14)





