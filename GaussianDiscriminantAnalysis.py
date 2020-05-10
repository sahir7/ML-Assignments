import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn import preprocessing
import math


X1 = np.genfromtxt("data/q4/q4x.dat")
X=preprocessing.scale(X1)
y = np.genfromtxt("data/q4/q4y.dat",dtype = str, delimiter=",")
Y=np.c_[y]
y1=(Y=='Alaska')*np.ones((len(Y),1))


def phi(Y):
    m = len(Y)
    return 1/m*np.sum((Y=='Alaska'))


def muh0(X,Y):
    num=np.dot(X.T,(Y=='Canada'))
   
    den=np.sum((Y=='Canada'))
    return (num/den)
u0=muh0(X,Y) 
    


def muh1(X,Y):
    num=np.dot(X.T,(Y=='Alaska'))
    
    den=np.sum((Y=='Alaska'))
    return (num/den)
u1=muh1(X,Y)    


U=np.empty((100,2))
for i in range(len(y)):
    if( y[i]=='Alaska'):
        U[i,0], U[i,1]=muh1(X,Y)
    else:
        U[i,0], U[i,1]=muh0(X,Y)
    
    


def sigma(X,U,Y):
    m=len(Y)
    ft=X-U
    st=ft.T
    return 1/m*np.dot(st,ft)


def P_Y(py):
    if(py=='Alaska'):
        return phi(Y)
    else:
        return 1 - phi(Y)


def Sigma0(X,Y,U):
    den=np.sum((Y=='Canada'))
    n1=(Y=='Canada')*(X-U)
    #n2=n1.T
    num=np.dot(n1.T,n1)
    return num/den



def Sigma1(X,Y,U):
    den=np.sum((Y=='Alaska'))
    n=(Y=='Alaska')*(X-U)
    num=np.dot(n.T,n)
    return num/den


def Z_Value(x0,x1):
    
    quadratic, linear = 0, 0
    quadratic += x0*x0*E_diff[0][0]
    quadratic += x0*x1*E_diff[0][1]
    quadratic += x1*x0*E_diff[1][0]
    quadratic += x1*x1*E_diff[1][1]
    linear += B[0,0]*x0
    linear += B[0,1]*x1
    return (quadratic+linear+C)


E=sigma(X,U,Y)
E0=Sigma0(X,Y,U)
E1=Sigma1(X,Y,U) 


#changing labels to int
x1_plot=X[:, :-1]
x2_plot=(-a[0,0]*x1_plot+b)/a[0,1]
y1=(Y=='Alaska')*np.ones((len(Y),1))

data=np.c_[X1_plot,X2_plot,y1]
x = np.sort(data[:, :-1])
y = data[:, -1]

Alaska=data[y==1]
Canada=data[y==0]


#Equation for linear decision boundary
b1=np.dot(u1.T,np.dot(np.linalg.inv(E),u1))
b2=np.dot(u0.T,np.dot(np.linalg.inv(E),u0))

b3=np.log(P_Y('Alaska'))-np.log(P_Y('Canada'))
b4=-1/2 * (b1-b2)

b=b4+b3

a=np.dot(u1.T,np.linalg.inv(E))-np.dot(u0.T,np.linalg.inv(E))



#for quadratic decision boundary equation
E1inv = np.linalg.inv(E1)
E0inv = np.linalg.inv(E0)

E_diff = E1inv-E0inv

E1det=np.linalg.det(E1)
E0det=np.linalg.det(E0)


a1=(E1inv-E0inv)

b1=np.dot(u1.T,E1inv.T)
b2=np.dot(u0.T,E0inv.T)

B=2*(b1-b2)

c1=2*np.log(math.sqrt(E0det)/math.sqrt(E1det))
c2=2*(np.log(P_Y('Alaska'))-np.log(P_Y('Canada')))

c3=np.dot(np.dot(u1.T,E1inv),u1)
c4=np.dot(np.dot(u0.T,E0inv),u0)

C=c1+c2+c3-c4


plt.scatter(Alaska[:,0],Alaska[:,1] ,  marker='v',label='Alaska')
plt.scatter(Canada[:, 0], Canada[:, 1],marker='o', label='Canada')
#Linear Decison boundary
plt.plot(x1_plot, x2_plot,label='Decision boundary',color='red')

#Quadratic Decision boundary
x0 = np.arange(-4, 5,1 )
x1 = np.arange(-4, 5,1)
x0,x1= np.meshgrid(x0, x1)
z=Z_Value(x0,x1)
plt.contour(x0, x1, z, levels=[0], cmap="Greys_r")

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

