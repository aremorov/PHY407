"""
Lab09 
Question-2
Author: Arya Kimiaghalam
Notes:
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft,irfft, fft2
import matplotlib.pyplot as plt
#from scipy.fftpack import dct,idct,dst,idst
###PART-A###:
"""
The first equation on E satistfies the boundary condition since for 
p,q = 0 or P,the sine terms will go to zero (their product) and therefore, 
the whole expression vanishes. That is what we want at the boundaries for E.

It is the same thing with current flux. The current flux should be zero at the
boundaries. The division by P in the arguments of sine functions in the 
decomposition guarantees that J will vanish whenever p or q becomes P or 0.

SAME STORY WITH THE OTHER TWO (EXPLAIN THIS MORE AND ASK TA).
"""
###PART-B###:

tao = 0.01  #subject to change.

def Jz(m,n,Lx,Ly,J0,w,t,P):
    """
    returns the flux for the (m,n) mode.
    """
    J = np.zeros((P,P),dtype=float)
    for i in range(1,P-1):
        for j in range(1,P-1):
            x = i*(Lx/P)
            y = j*(Ly/P)
            J[i,j] = J0*(np.sin((np.pi)*m*x/Lx)*np.sin((np.pi)*n*y/Ly)*np.sin(w*t))
    return J
P = 32
Lx = 1
Ly = 1
J0 = 1
m = 1
n = 1
c = 1
w = 3.75
def fc_update(E,X,Y,J):
    """
    Updates Fourier coefficients of everything except flux
    since you can update that separately.
    """
    E_new = np.zeros((len(E),len(E)),dtype=complex)
    X_new = np.zeros((len(E),len(E)),dtype=complex)
    Y_new = np.zeros((len(E),len(E)),dtype=complex)
    Dx = (np.pi)*c*tao/(2*Lx)
    Dy = (np.pi)*c*tao/(2*Ly)
    for p in range(len(E)):
        for q in range(len(E)):
            k1 = ((((1-(p**2)*(Dx**2)-(q**2)*(Dy**2))*E[p,q]) + 2*q*Dy*X[p,q] + 2*p*Dx*Y[p,q] + tao*J[p,q])/(1+(p**2)*(Dx**2)+(q**2)*(Dy**2)))
            k2 = X[p,q] - q*Dy*(k1+E[p,q])
            k3 = Y[p,q] - p*Dx*(k1+E[p,q])
            E_new[p,q] = k1
            X_new[p,q] = k2
            Y_new[p,q] = k3
    return [E_new,X_new,Y_new]

#
#
def dct(y):    
    N = len(y)    
    y2 = np.empty(2*N,float)    
    y2[:N] = y[:]    
    y2[N:] = y[::-1]    
    c = rfft(y2)    
    phi = np.exp(-1j*np.pi*np.arange(N)/(2*N))    
    return np.real(phi*c[:N])

def idct(a):    
    N = len(a)    
    c = np.empty(N+1,complex)    
    phi = np.exp(1j*np.pi*np.arange(N)/(2*N))    
    c[:N] = phi*a    
    c[N] = 0.0    
    return irfft(c)[:N]



def dst(y):    
    N = len(y)    
    y2 = np.empty(2*N,float)    
    y2[0] = y2[N] = 0.0    
    y2[1:N] = y[1:]    
    y2[:N:-1] = -y[1:]    
    a = -np.imag(rfft(y2))[:N]    
    a[0] = 0.0    
    return a

def idst(a):    
    N = len(a)    
    c = np.empty(N+1,complex)    
    c[0] = c[N] = 0.0    
    c[1:N] = -1j*a[1:]    
    y = irfft(c)[:N]    
    y[0] = 0.0    
    return y


#######################################
def sinsin(A):
    l = len(A[0])
    sin_1 = np.zeros((l,l))
    sin_2 = np.zeros((l,l))
    #first row transforms:
    for i in range(l):
        sin_1[i] = dst(A[i])
    #Second transforms:
    for j in range(l):
        sin_2[:,j] = dst(sin_1[:,j])
    return sin_2

def sincos(A):
    l = len(A[0])
    sin_1 = np.zeros((l,l))
    cos_2 = np.zeros((l,l))
    #first row sin transform:
    for i in range(l):
        sin_1[i] = dst(A[i])
    #second cos transform for columns:
    for j in range(l):
        cos_2[:,j] = dct(sin_1[:,j])
    return cos_2

def cossin(A):
    l = len(A[0])
    cos_1 = np.zeros((l,l))
    sin_2 = np.zeros((l,l))
    #first row sin transform:
    for i in range(l):
        cos_1[i] = dct(A[i])
    #second cos transform for columns:
    for j in range(l):
        sin_2[:,j] = dst(cos_1[:,j])
    return sin_2

def i_sinsin(B):
    l = len(B[0])
    sin_1 = np.zeros((l,l))
    sin_2 = np.zeros((l,l))
    #first inverse transform along the columns:
    for i in range(l):
        sin_2[:,i] = idst(B[:,i])
    for j in range(l):
        sin_1[j] = idst(sin_2[j])
    return sin_1

def i_sincos(B):
    l = len(B[0])
    cos_2 = np.zeros((l,l))
    sin_1 = np.zeros((l,l))
    #first inverse transform of columns:
    for i in range(l):
        cos_2[:,i] = idct(B[:,i])
    for j in range(l):
        sin_1[j] = idst(cos_2[j])
    return sin_1


def i_cossin(B):
    l = len(B[0])
    sin_2 = np.zeros((l,l))
    cos_1 = np.zeros((l,l))
    #first inverse transformation of columns:
    for i in range(l):
        sin_2[:,i] = idst(B[:,i])
    for j in range(l):
        cos_1[j] = idct(sin_2[j])
    return cos_1

def EH_update(w,t):
    """
    updating E, Hx and Hy for time t.
    """
    N = t/tao
    num = 0
    E_inter = np.zeros((P,P))
    Hx_inter = np.zeros((P,P))
    Hy_inter = np.zeros((P,P))
    while num < N:
        J = Jz(m,n,Lx,Ly,J0,w,num*tao,P)
        J_hats = sinsin(J)
        E_hat_inter = sinsin(E_inter)
        Hx_hat_inter = sincos(Hx_inter)
        Hy_hat_inter = cossin(Hy_inter)
        
        E_hat_upd = fc_update(E_hat_inter,Hx_hat_inter,Hy_hat_inter,J_hats)[0]
        Hx_hat_upd = fc_update(E_hat_inter,Hx_hat_inter,Hy_hat_inter,J_hats)[1]
        Hy_hat_upd = fc_update(E_hat_inter,Hx_hat_inter,Hy_hat_inter,J_hats)[2]
        
        E_inter = i_sinsin(E_hat_upd)
        Hx_inter = i_sincos(Hx_hat_upd)
        Hy_inter = i_cossin(Hy_hat_upd)
        num = num + 1
    return [E_inter,Hx_inter,Hy_inter]
    
Hx = []
Hy = []
Ez = []
times = np.arange(0,20,10*tao)
for i in times:
    Hx.append(EH_update(w,i)[1][31,15])
    Hy.append(EH_update(w,i)[2][15,0])
    Ez.append(EH_update(w,i)[0][15,15])

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






#plt.imshow(EH_update(0.1)[0])
#plt.title("Ez")
#plt.colorbar()
#plt.show()
#
#plt.imshow(EH_update(0.1)[1])
#plt.title("Hx")
#plt.colorbar()
#plt.show()
#
#plt.imshow(EH_update(0.1)[2])
#plt.title("Hy")
#plt.colorbar()
#plt.show()





#lst = np.array([0,1,2,0])
#jojo = np.zeros((4,4))
#for i in range(3):
#    jojo[i:,] = lst
#
#
#    
#print(i_sinsin(sinsin(np.array(jojo))))
#print("################")
#print(i_sincos(sincos(np.array(jojo))))
#print("################")
#print(i_cossin(cossin(np.array(jojo))))
