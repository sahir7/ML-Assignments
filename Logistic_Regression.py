#!/usr/bin/env python
# coding: utf-8

# In[624]:


import numpy as np
import pandas as pd 
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn import preprocessing


# In[625]:



x_1,x_2 = np.genfromtxt("data/q3/logisticX.csv", delimiter=',').transpose()
y_= np.genfromtxt("data/q3/logisticY.csv")
theta=np.zeros((3,1))
it=10
x__1=preprocessing.scale(x_1)
x__2=preprocessing.scale(x_2)
x=np.c_[np.ones((len(x_))),x__1,x__2]
y=np.c_[y_]
x1=x.T
tolerance=.000000001
l_history=[]


# In[626]:


def hypothesis(x,y,theta):
    z=(np.dot(x,theta))
    return 1.0/(1+np.exp(-z))


# In[627]:


def log_likelihood(x,y,theta):
    hypo=hypothesis(x,y,theta)
    f=np.multiply(y,np.log(hypo))
    s=np.ones((len(y),1))-y
    s1=np.log(np.ones((len(x),1))-hypo) 
    l=np.sum(f+np.multiply(s,s1))
    return np.exp(l)


# In[628]:


def gradient(x,y,theta,x1):
    g=(np.dot(x1,y-hypothesis(x,y,theta)))
    return g


# In[629]:


def hessian(x,y,theta,x1):
    hypo=hypothesis(x,y,theta)
    hypo1=np.ones((len(x),1))-hypo
    d=np.multiply(hypo,hypo1).T
    s=np.dot(np.diag(d[0]),x)
    h=np.dot(x1,s)
    return np.linalg.inv(h)


# In[630]:


for i in range(it):
    h=hessian(x,y,theta,x1)
    g=gradient(x,y,theta,x1)
    a= theta + np.dot(h,g)
    theta=np.copy(a)
    l_history.append(log_likelihood(x,y,theta))
    
print(theta)    


# In[631]:


l=np.array(l_history)


# In[632]:


x2 = np.array([min(x[:,0])-3 , max(x[:,0]) +3])
y_values=-(theta[0]+theta[1]*x2)/theta[2]
data=np.c_[x__1,x__2,y]
X = data[:, :-1]
Y = data[:, -1]
admitted = data[Y== 1]
not_admitted = data[Y == 0]
plt.scatter(admitted[:, 0], admitted[:, 1], s=10, label='Admitted')
plt.scatter(not_admitted[:, 0], not_admitted[:, 1], s=10, label='Not Admitted')
plt.plot(x2, y_values, label='Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')

plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




