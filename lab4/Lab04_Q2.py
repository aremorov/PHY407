import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin
#we convert everything into SI just to be safe:
L = 5*(10**-10)  #width of the well in meters.
eV = 1.6*(10**-19)  #electron volts in joules. 
def H(m,n,L):
    """
    This function calculates the [m,n] entry of the hamiltonian matrix of the 
    Schrodinger Equation for specific m and n values given the width of the asymmetric well.
    The algorithm an the relevant integrals used are in page 249 of the Text Book.
    """
    h = (1.05)*(10**-34) # planks constantin J/s
    M = 9.11*(10**-31)  # mass in kg
    a = 1.6*(10**-18) #a_value.
    #L is measured in angstroms here.
    c_1 = ((np.pi**2)*(n**2)*(h**2))/((L**3)*M)
    c_2 = ((2*a)/(L**2))
    first_int = []
    second_int = []
    #taking care of the first integral.
    if m == n:
        first_int.append(L/2)
    else:
        first_int.append(0)
    #taking care of the second integral.
    if m == n:
        second_int.append((L**2)/4)
    elif m%2 == n%2:
        second_int.append(0)
    else:
        second_int.append(((m*n)/(((m**2)-(n**2))**2))*(((2*L)/np.pi)**2)*(-1))
    return ((c_1*first_int[0])+(c_2*second_int[0]))

def H_matrix(size,L):
    """
    returns the Hamiltonian matrix os a specified size given the 
    width of the asymmetric well.
    """
    matrix = np.zeros((size,size))
    for m in range(1, size+1):  #assigning values to the Hamiltonian.
        for n in range(1, size+1):
            matrix[m-1, n-1] = H(m,n,L)
    return matrix

#Eigen values (i.e. energies for n=10 and n=100)
energies_1,eig_v_1 = lin.eigh(H_matrix(10,L))  #first 10.
energies_2,eig_v_2 = lin.eigh(H_matrix(100,L)) #first 100.
#printing out the energy values of the first 10 states.
for i in range(10):
    print("The energy level of state "+str(i)+" is",energies_1[i]*(eV**-1), "eV")
#printing out the difference in value for the first 10 entries of H_10 and H_100 as error.
for i in range(10):
    print("Error for E"+str(i)+" is",abs(energies_1[i]-energies_2[i])*(eV**-1),"eV")

def psi(n,x):
    """
    returns the value of the wave function of a particular level at a specified 
    position x, in meters.
    """
    values = []
    coeffs = eig_v_2[:,n]   #we choose columns of that matrix to be the eigenvector because otherwise
                            #the ground state would be more probable towards x=L, which has a higher potential.
                            #This does not make sense so we choose the latter.
    for i in range(100):
        values.append(coeffs[i]*np.sin(np.pi*(i+1)*x/L))  #Fourier Series.
    return sum(values)

def Norm_psi_2(n):
    """
    Returns the plot a normalized set of data for different wave functions probability amplitude.
    Uses trapazoidal method for normalization.
    """
    #first we find the area under the curve to normalize it:
    h = L/100
    lst = []
    for k in range(1,101):
        lst.append(psi(n,(k*h))**2)
    integral = h*((0.5*psi(n,0)**2+0.5*psi(n,L)**2)+sum(lst))
    norm_fac = 1/(integral)
    #return np.array(lst)*norm_fac
    plt.plot(np.arange(h,L+h,h),np.array(lst)*norm_fac,linewidth=3,label="n = "+str(n))
    plt.title("Normalized Wave Function of the System",fontsize=14)
    plt.xlabel("X (m)",fontsize=14)
    plt.ylabel("\u03C8"+" (x)",fontsize=14)

#plotting the first threee states:
Norm_psi_2(0)
Norm_psi_2(1)
Norm_psi_2(2)
plt.legend()

    

