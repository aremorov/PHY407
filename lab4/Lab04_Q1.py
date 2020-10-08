from numpy import array,empty,copy,random,mean,dot,arange,zeros,log,e,pi,real
import numpy as np
from numpy.linalg import solve
from time import time
import matplotlib.pyplot as plt
from cmath import phase

###Part-a:
def gaussian_elimination(A,v):
    """
    This Function solves a linear system of equations
    by Gaussian elimination.
    Credit: Mark Newman.
    """
    N = len(v)

    # Gaussian elimination
    for m in range(N):
    
        # Divide by the diagonal element
        div = A[m,m]
        A[m,:] /= div
        v[m] /= div
    
        # Now subtract from the lower rows
        for i in range(m+1,N):
            mult = A[i,m]
            A[i,:] -= mult*A[m,:]
            v[i] -= mult*v[m]
    
    # Backsubstitution
    x = empty(N,float)
    for m in range(N-1,-1,-1):
        x[m] = v[m]
        for i in range(m+1,N):
            x[m] -= A[m,i]*x[i]
    return x





def PartialPivot(A,v):
    """
    Uses Partial pivot technique in the process of Gaussian
    elimination to prevent failure in case of a zero leading element.
    A: s square matrix.
    v: values matrix (right hand side of all the equations listed as a vertical vector)
    """

    N = len(v)
    
    # Gaussian elimination
    for m in range(N):
        heads = A[::,m]           #collecting leading elements of the m-th stel in the elimination to ultimately select a good candidate. 
        abs_heads = list(abs(heads))
        winning = abs_heads.index(max(abs_heads))
        if heads[m] == 0:
            A[m, :], A[winning, :] = copy(A[winning, :]), copy(A[m, :])
            v[m], v[winning] = copy(v[winning]), copy(v[m])
        else:
            pass
        # Divide by the diagonal element
        div = A[m,m]
        A[m,:] /= div
        v[m] /= div
    
        # Now subtract from the lower rows
        for i in range(m+1,N):
            mult = A[i,m]
            A[i,:] -= mult*A[m,:]
            v[i] -= mult*v[m]
    
    # Backsubstitution
    x = empty(N,float)
    for m in range(N-1,-1,-1):
        x[m] = v[m]
        for i in range(m+1,N):
            x[m] -= A[m,i]*x[i]
    return x


##Part-b:
def method_analyzer(N):
    """
    finds the processing time and error for different methods of solving
    linear equations of random characteristics. Methods include: Guassian elimination, Partial piviting
    and LU-decomposition for a range of matrix sizes (5,N)
    N: a positive integer.
    """
    #we had to generate copies of the coefficient matrix since each operation
    #affects the original matrix and when the next line uses the newly changed matrix
    #it would no longer use the same matrix as previous steps and things will get
    #mixed up. This is the safe route.
    gaussian_time = []
    pivot_time = []
    LU_time = []
    gaussian_error = []
    pivot_error = []
    LU_error = []
    for n in range(5,N+1):
        matrix_A = random.rand(n,n)
        A_1 = copy(matrix_A)
        A_2 = copy(matrix_A)
        A_3 = copy(matrix_A)
        A_4 = copy(matrix_A)
        A_5 = copy(matrix_A)
        matrix_v = random.rand(1,n)[0]
        v_1 = copy(matrix_v)
        v_2 = copy(matrix_v)
        v_3 = copy(matrix_v)
        v_4 = copy(matrix_v)
        v_5 = copy(matrix_v)
        s_1 = time()
        x = gaussian_elimination(matrix_A,matrix_v)
        e_1 = time()
        gaussian_time.append(e_1-s_1)
        gaussian_error.append(mean(abs(v_1-dot(A_1,x))))
        s_2 = time()
        y = PartialPivot(A_2,v_2)
        e_2 = time()
        pivot_time.append(e_2-s_2)
        pivot_error.append(mean(abs(v_3-dot(A_3,y))))
        s_3 = time()
        z = solve(A_4,v_4)
        e_3 = time()
        LU_time.append(e_3-s_3)
        LU_error.append(mean(abs(v_5-dot(A_5,z))))
    return [gaussian_time,pivot_time,LU_time,gaussian_error,pivot_error,LU_error]
        
#generating plots:       
plt.semilogy(method_analyzer(200)[5],color='r',label="LU Decomposition")        
plt.semilogy(method_analyzer(200)[4],color='b',label="Partial Pivoting") 
plt.semilogy(method_analyzer(200)[3],color='green',label="Gaussian Elimination")
plt.title("Error of Different Methods for Different System Sizes",fontsize=14)
plt.xlabel("Number of Variables Involved",fontsize=14)
plt.ylabel("Log(error)",fontsize=14)    
plt.legend()
    
plt.plot(method_analyzer(200)[2],color='r',label='LU Decomposition')        
plt.plot(method_analyzer(200)[1],color='b',label="Partial Pivoting") 
plt.plot(method_analyzer(200)[0],color='green',label="Gaussian Elimination") 
plt.title("Processing time of Different Methods for Different System Sizes",fontsize=14)
plt.xlabel("Number of Variables Involved",fontsize=14)
plt.ylabel("Time in seconds",fontsize=14)    
plt.legend()

##Part-c:
r_1 = r_3 = r_5 = 1000
r_2 = r_4 = r_6 = 2000
c_1 = 10**-6
c_2 = (0.5)*(10**-6)
x_p = 3
w = 1000
coef_matrix = zeros((3,3),dtype=complex)
v_matrix = zeros((1,3),dtype=complex) 
coef_matrix[0,0] = ((1/r_1)+(1/r_4)+1j*w*c_1)
coef_matrix[0,1] = -1j*w*c_1
coef_matrix[0,2] =  0
coef_matrix[1,0] = -1j*w*c_1
coef_matrix[1,1] = ((1/r_2)+(1/r_5)+1j*w*c_1+1j*w*c_2)
coef_matrix[1,2] = -1j*w*c_2
coef_matrix[2,0] = 0
coef_matrix[2,1] = -1j*w*c_2
coef_matrix[2,2] = ((1/r_3)+(1/r_6)+1j*w*c_2)
v_matrix[0,0] = x_p/r_1
v_matrix[0,1] = x_p/r_2
v_matrix[0,2] = x_p/r_3

#
def PP_complex(A,v):
    """
    Uses Partial pivot technique in the process of Gaussian
    elimination to prevent failure in case of a zero leading element.
    A: s square matrix.
    v: values matrix (right hand side of all the equations listed as a vertical vector).
    NOTE: This function is geared towards receiving complex valued matrices.
    """

    N = len(v)
    
    # Gaussian elimination
    for m in range(N):
        heads = A[::,m]
        abs_heads = list(abs(heads))
        winning = abs_heads.index(max(abs_heads))
        if heads[m] == 0:
            A[m, :], A[winning, :] = copy(A[winning, :]), copy(A[m, :])
            v[m], v[winning] = copy(v[winning]), copy(v[m])
        else:
            pass
        # Divide by the diagonal element
        div = A[m,m]
        A[m,:] = A[m,:]/div
        (v[m]) = (v[m])/div
    
        # Now subtract from the lower rows
        for i in range(m+1,N):
            mult = A[i,m]
            A[i,:] -= mult*A[m,:]
            (v[i]) -= mult*v[m]
    
    # Backsubstitution
    x = empty(N,complex)  #receiving complex values instead of float.
    for m in range(N-1,-1,-1):
        x[m] = (v[m])
        for i in range(m+1,N):
            x[m] -= A[m,i]*x[i]
    return x

#generating copies of the coefficient matricies.      
coef_matrix_copy = copy(coef_matrix)       
v_matrix_copy = copy(v_matrix[0]) 
solution = PP_complex(coef_matrix_copy,v_matrix_copy)
print("x1= "+str(solution[0])+"x2= "
      +str(solution[1])+"x3= "+str(solution[2]))    
  
def V(X,w,t):
    """
    returns the potential at a time t (in seconds) given the set X=(x1,x2,x3)
    and w.
    It returns a list containing values, amplitudes and phases (in radians).
    """
    results = []
    amplitudes = []
    phases = []
    for x in X:
        results.append((x)*(e**(1j*w*t)))
        amplitudes.append(abs(x))
        phases.append(phase((x)*(e**(1j*w*t))))
    return [results,amplitudes,phases]
    
#printing results for t=0:
print(V(solution,1000,0)[1], "These are amplitudes at of V1 V2 and V3 t=0.")
print(V(solution,1000,0)[2],"These are phases at of V1 V2 and V3 t=0.")

def V_real_value(t_max,w,x):
    """
    returns an array of V values over time.
    Note that the arrays only contain the real part of V.
    """
    points = arange(0,t_max,0.05)   #chooseing time step of 0.05 seconds because of aliacing problems. 

    values= x*np.exp(1j*w*points)
    return [np.real(values),points]

#ploting the first series of V:
plt.plot(V_real_value(3,1000,solution[2])[1],V_real_value(3,1000,solution[2])[0],label="V_3",linewidth=3)
plt.plot(V_real_value(3,1000,solution[1])[1],V_real_value(3,1000,solution[1])[0],label="V_2",linewidth=3)
plt.plot(V_real_value(3,1000,solution[0])[1],V_real_value(3,1000,solution[0])[0],label="V_1",linewidth=3)
plt.title("Voltage vs. Time",fontsize=14)
plt.xlabel("Time (s)",fontsize=14)
plt.ylabel("Voltage (volts)",fontsize=14)
plt.legend()

#generating the coefficient matrix for the new set of factors (introduction of an inductor):
coef_matrix_2 = zeros((3,3),dtype=complex)
v_matrix_2 = zeros((1,3),dtype=complex) 
coef_matrix_2[0,0] = ((1/r_1)+(1/r_4)+1j*w*c_1)
coef_matrix_2[0,1] = -1j*w*c_1
coef_matrix_2[0,2] =  0
coef_matrix_2[1,0] = -1j*w*c_1
coef_matrix_2[1,1] = ((1/r_2)+(1/r_5)+1j*w*c_1+1j*w*c_2)
coef_matrix_2[1,2] = -1j*w*c_2
coef_matrix_2[2,0] = 0
coef_matrix_2[2,1] = -1j*w*c_2
coef_matrix_2[2,2] = ((1/r_3)+(1/(1j*r_6))+1j*w*c_2)
v_matrix_2[0,0] = x_p/r_1
v_matrix_2[0,1] = x_p/r_2
v_matrix_2[0,2] = x_p/r_3

solution_2 = PP_complex(coef_matrix_2,v_matrix_2[0]) #solution for the second set of variables and factors.
#generating plots of voltages:
plt.plot(V_real_value(3,1000,solution_2[2])[1],V_real_value(3,1000,solution_2[2])[0],label="V_3",linewidth=3)
plt.plot(V_real_value(3,1000,solution_2[1])[1],V_real_value(3,1000,solution_2[1])[0],label="V_2",linewidth=3)
plt.plot(V_real_value(3,1000,solution_2[0])[1],V_real_value(3,1000,solution_2[0])[0],label="V_1",linewidth=3)
print(V(solution_2,1000,0)[1], "These are amplitudes at of V1 V2 and V3 t=0.")
print(V(solution_2,1000,0)[2],"These are phases at of V1 V2 and V3 t=0.")
plt.title("Voltage vs. Time after the Introduction of an Inductor",fontsize=14)
plt.xlabel("Time (s)",fontsize=14)
plt.ylabel("Voltage (volts)",fontsize=14)
plt.legend()
#returning the amplitudes and phases of voltages at t=0 in the second setting:
print(V(solution_2,1000,0)[1], "These are amplitudes at of V1 V2 and V3 after the intriduction of an inductor at t=0.")
print(V(solution_2,1000,0)[2],"These are phases at of V1 V2 and V3 after the intriduction of an inductor at t=0.")