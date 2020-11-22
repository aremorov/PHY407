import numpy as np
import matplotlib.pyplot as plt

# Constants
m = 9.1094e-31     # Mass of electron (kg)
hbar = 1.0546e-34  # Planck's constant over 2*pi (J*s)
P = 1024             #number of cells
L = 1e-8          #lengthscale (m)
a = L/P            #length step size (m)
tau = 1e-18       #timestep (s)
sigma = L/25       #gaussian width parameter (m)
k = 500/L          #wavenumber (m^(-1))
x = np.linspace(-L/2, L/2, P-1) #spatial domain
x0 = L/3 #gaussian is centered about x0 (s)
omega = 3e15       #harmonic oscillator frequency (s^(-1))
V0 = 6e-17         #reference potential (J)
x1 = L/4           #location of potential V3 (m)
#Potential Functions:
def V1(x): #square well
    return 0

def V2(x): #harmonic oscillator
    return (1/2)*m*omega**2 * x**2

def V3(x): #double well
    return V0*(x**2 / x1**2 - 1)**2

"""

#create and populate discretized Hamiltonian:
V = V1 #use square potential
H = np.zeros([P-1,P-1]) 
A = -(hbar**2)/(2*m*a**2)
Sup = A*np.eye(P-1,k=1) #A on 1st super-diagonal
Sub = A*np.eye(P-1,k=-1) #A on 1st sub-diagonal
for i in range(1,P):
    H[i-1,i-1] = V(i*a - L/2) - 2*A #B elements along diagonal
H = H + Sup + Sub
        
#L, R matrices:
Lm = np.eye(P-1) + (1j)*(tau/(2*hbar))*H
Rm = np.eye(P-1) - (1j)*(tau/(2*hbar))*H
    
#initial wavefunction
psi0 = 1/(np.sqrt(sigma * np.sqrt(2*np.pi))) #normalization constant
psi = psi0*np.exp(-(x-x0)**2 / (4*sigma**2) + (1j)*k*x)

#Save plot of |psi|^2 at t=0
plt.plot(x/L,np.real(psi*np.conj(psi)))
plt.xlabel("x/L")
plt.ylabel("Probability Density")
plt.title(r"$|\psi(x,t=0)|^2$")
plt.savefig("psi_t0_b.pdf")
plt.clf()



#Defining important diagnostic functions
def X_exp(psi): #get expected position
    return np.sum(np.conj(psi)*x*psi)*a

def E(psi): #get energy
    return np.sum(np.dot(np.dot(np.conj(psi), H), psi))*a

def norm(psi): #get normalization
    return np.sum(np.conj(psi)*psi)*a

#Running the simulation:
N = 3000 #number of time steps
psi_list = [] #storing time seperated absolute wavefunction profiles
X_list = [] #store expected position
E_list = [] #store energy
N_list = [] #store normalization

#running the Crank-Nicolson Method:
for i in range(N):
    
    if(i == int(N/4)):
        psi14 = psi
    
    if(i == int(N/2)):
        psi12 = psi
        
    if(i == int(3*N/4)):
        psi34 = psi
    
    v = np.dot(Rm,psi)
    psi = np.linalg.solve(Lm, v)
    
    X_list.append(X_exp(psi))
    E_list.append(E(psi))
    N_list.append(norm(psi))
    

#Plotting results:
t = np.linspace(0,N*tau,N)

#Save plot of |psi|^2 at t=T/4
plt.plot(x/L,np.real(psi14*np.conj(psi14)))
plt.xlabel("x/L")
plt.ylabel("Probability Density")
plt.title(r"$|\psi(x,t=T/4)|^2$")
plt.savefig("psi_t14_b.pdf")
plt.clf()

#Save plot of |psi|^2 at t=T/2
plt.plot(x/L,np.real(psi12*np.conj(psi12)))
plt.xlabel("x/L")
plt.ylabel("Probability Density")
plt.title(r"$|\psi(x,t=T/2)|^2$")
plt.savefig("psi_t12_b.pdf")
plt.clf()

#Save plot of |psi|^2 at t=3T/4
plt.plot(x/L,np.real(psi34*np.conj(psi34)))
plt.xlabel("x/L")
plt.ylabel("Probability Density")
plt.title(r"$|\psi(x,t=3T/4)|^2$")
plt.savefig("psi_t34_b.pdf")
plt.clf()

#Save plot of |psi|^2 at t=T
plt.plot(x/L,np.real(psi*np.conj(psi)))
plt.xlabel("x/L")
plt.ylabel("Probability Density")
plt.title(r"$|\psi(x,t=T)|^2$")
plt.savefig("psi_t1_b.pdf")
plt.clf()

#Normalization
plt.plot(t/tau, np.real(np.round_(N_list, 10)))
plt.xlabel(r"$t/\tau$")
plt.ylabel(r"$<\psi|\psi>$")
plt.title("Normalization")
plt.savefig("N_a.pdf")
plt.clf()

#Energy 
plt.plot(t/tau, np.real(np.round_(E_list, 21)))
plt.xlabel(r"$t/\tau$")
plt.ylabel("E (J)")
plt.title("Energy")
plt.savefig("E_a.pdf")
plt.clf()

#Expectation
plt.plot(t/tau, np.real(np.round_(X_list, 12))/L)
plt.xlabel(r"$t/\tau$")
plt.ylabel("<X>/L")
plt.title("Expected Position")
plt.savefig("X_a.pdf")
plt.clf()


#______________________________________________________________________________

#______________________________________________________________________________

#______________________________________________________________________________


#create and populate discretized Hamiltonian:
V = V2 #use square potential
H = np.zeros([P-1,P-1]) 
A = -(hbar**2)/(2*m*a**2)
Sup = A*np.eye(P-1,k=1) #A on 1st super-diagonal
Sub = A*np.eye(P-1,k=-1) #A on 1st sub-diagonal
for i in range(1,P):
    H[i-1,i-1] = V(i*a - L/2) - 2*A #B elements along diagonal
H = H + Sup + Sub
        
#L, R matrices:
Lm = np.eye(P-1) + (1j)*(tau/(2*hbar))*H
Rm = np.eye(P-1) - (1j)*(tau/(2*hbar))*H
    
#initial wavefunction
psi0 = 1/(np.sqrt(sigma * np.sqrt(2*np.pi))) #normalization constant
psi = psi0*np.exp(-(x-x0)**2 / (4*sigma**2) + (1j)*k*x)


#Save plot of |psi|^2 at t=0
plt.plot(x/L,np.real(psi*np.conj(psi)))
plt.xlabel("x/L")
plt.ylabel("Probability Density")
plt.title(r"$|\psi(x,t=0)|^2$")
plt.savefig("psi_t0_d.pdf")
plt.clf()


#Defining important diagnostic functions
def X_exp(psi): #get expected position
    return np.sum(np.conj(psi)*x*psi)*a

def E(psi): #get energy
    return np.sum(np.dot(np.dot(np.conj(psi), H), psi))*a

def norm(psi): #get normalization
    return np.sum(np.conj(psi)*psi)*a

#Running the simulation:
N = 4000 #number of time steps
psi_list = [] #storing time seperated absolute wavefunction profiles
X_list = [] #store expected position
E_list = [] #store energy
N_list = [] #store normalization

#running the Crank-Nicolson Method:
for i in range(N):
    
    if(i == int(N/4)):
        psi14 = psi
    
    if(i == int(N/2)):
        psi12 = psi
        
    if(i == int(3*N/4)):
        psi34 = psi
    
    v = np.dot(Rm,psi)
    psi = np.linalg.solve(Lm, v)
    
    X_list.append(X_exp(psi))
    E_list.append(E(psi))
    N_list.append(norm(psi))
    

#Plotting results:
t = np.linspace(0,N*tau,N)

#Save plot of |psi|^2 at t=T/4
plt.plot(x/L,np.real(psi14*np.conj(psi14)))
plt.xlabel("x/L")
plt.ylabel("Probability Density")
plt.title(r"$|\psi(x,t=T/4)|^2$")
plt.savefig("psi_t14_c.pdf")
plt.clf()

#Save plot of |psi|^2 at t=T/2
plt.plot(x/L,np.real(psi12*np.conj(psi12)))
plt.xlabel("x/L")
plt.ylabel("Probability Density")
plt.title(r"$|\psi(x,t=T/2)|^2$")
plt.savefig("psi_t12_c.pdf")
plt.clf()

#Save plot of |psi|^2 at t=3T/4
plt.plot(x/L,np.real(psi34*np.conj(psi34)))
plt.xlabel("x/L")
plt.ylabel("Probability Density")
plt.title(r"$|\psi(x,t=3T/4)|^2$")
plt.savefig("psi_t34_c.pdf")
plt.clf()

#Save plot of |psi|^2 at t=T
plt.plot(x/L,np.real(psi*np.conj(psi)))
plt.xlabel("x/L")
plt.ylabel("Probability Density")
plt.title(r"$|\psi(x,t=T)|^2$")
plt.savefig("psi_t1_c.pdf")
plt.clf()

#Expectation
plt.plot(t/tau, np.real(np.round_(X_list, 12))/L)
plt.xlabel(r"$t/\tau$")
plt.ylabel("<X>/L")
plt.title("Expected Position")
plt.savefig("X_c.pdf")
plt.clf()


"""

#______________________________________________________________________________
#______________________________________________________________________________
#______________________________________________________________________________


#create and populate discretized Hamiltonian:
V = V3 #use square potential
H = np.zeros([P-1,P-1]) 
A = -(hbar**2)/(2*m*a**2)
Sup = A*np.eye(P-1,k=1) #A on 1st super-diagonal
Sub = A*np.eye(P-1,k=-1) #A on 1st sub-diagonal
for i in range(1,P):
    H[i-1,i-1] = V(i*a - L/2) - 2*A #B elements along diagonal
H = H + Sup + Sub
        
#L, R matrices:
Lm = np.eye(P-1) + (1j)*(tau/(2*hbar))*H
Rm = np.eye(P-1) - (1j)*(tau/(2*hbar))*H
    
#initial wavefunction
psi0 = 1/(np.sqrt(sigma * np.sqrt(2*np.pi))) #normalization constant
psi = psi0*np.exp(-(x-x0)**2 / (4*sigma**2) + (1j)*k*x)

#Save plot of |psi|^2 at t=0
plt.plot(x/L,np.real(psi*np.conj(psi)))
plt.xlabel("x/L")
plt.ylabel("Probability Density")
plt.title(r"$|\psi(x,t=0)|^2$")
plt.savefig("psi_t0_d.pdf")
plt.clf()



#Defining important diagnostic functions
def X_exp(psi): #get expected position
    return np.sum(np.conj(psi)*x*psi)*a

def E(psi): #get energy
    return np.sum(np.dot(np.dot(np.conj(psi), H), psi))*a

def norm(psi): #get normalization
    return np.sum(np.conj(psi)*psi)*a

#Running the simulation:
N = 6000 #number of time steps
psi_list = [] #storing time seperated absolute wavefunction profiles
X_list = [] #store expected position
E_list = [] #store energy
N_list = [] #store normalization

#running the Crank-Nicolson Method:
for i in range(N):
    
    if(i == int(N/4)):
        psi14 = psi
    
    if(i == int(N/2)):
        psi12 = psi
        
    if(i == int(3*N/4)):
        psi34 = psi
    
    v = np.dot(Rm,psi)
    psi = np.linalg.solve(Lm, v)
    
    X_list.append(X_exp(psi))
    E_list.append(E(psi))
    N_list.append(norm(psi))
    

#Plotting results:
t = np.linspace(0,N*tau,N)

#Save plot of |psi|^2 at t=T/4
plt.plot(x/L,np.real(psi14*np.conj(psi14)))
plt.xlabel("x/L")
plt.ylabel("Probability Density")
plt.title(r"$|\psi(x,t=T/4)|^2$")
plt.savefig("psi_t14_d.pdf")
plt.clf()

#Save plot of |psi|^2 at t=T/2
plt.plot(x/L,np.real(psi12*np.conj(psi12)))
plt.xlabel("x/L")
plt.ylabel("Probability Density")
plt.title(r"$|\psi(x,t=T/2)|^2$")
plt.savefig("psi_t12_d.pdf")
plt.clf()

#Save plot of |psi|^2 at t=3T/4
plt.plot(x/L,np.real(psi34*np.conj(psi34)))
plt.xlabel("x/L")
plt.ylabel("Probability Density")
plt.title(r"$|\psi(x,t=3T/4)|^2$")
plt.savefig("psi_t34_d.pdf")
plt.clf()

#Save plot of |psi|^2 at t=T
plt.plot(x/L,np.real(psi*np.conj(psi)))
plt.xlabel("x/L")
plt.ylabel("Probability Density")
plt.title(r"$|\psi(x,t=T)|^2$")
plt.savefig("psi_t1_d.pdf")
plt.clf()

#Expectation
plt.plot(t/tau, np.real(np.round_(X_list, 12))/L)
plt.xlabel(r"$t/\tau$")
plt.ylabel("<X>/L")
plt.title("Expected Position")
plt.savefig("X_d.pdf")
plt.clf()


