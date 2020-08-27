#!/usr/bin/env python
# coding: utf-8

# In[221]:



from mpl_toolkits import mplot3d
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt

#fig = plt.figure()
#ax = plt.axes(projection="3d")

from sklearn import preprocessing
x = np.loadtxt("data/q1/linearX.csv",
y= np.loadtxt("data/q1/linearY.csv")
alpha = 0.1 
#current_theta =  np.array([[0,0]]).T


current_theta= np.zeros((2, 1))

X_scaled = preprocessing.scale(x)

X = np.c_[np.ones(len(x)),X_scaled]

Y = np.c_[y]

X_1=np.c_[X_scaled].T

cost_history=[]
theta0=[]
theta1=[]
previous_theta=np.array([[-.000002],[-.000002]])


# In[222]:


def costfunction(X,Y,current_theta):
    j=1/2*len(Y)* np.square(np.sum(Y-np.dot(X,current_theta)))
    return j


# In[223]:


tolerance=0.000001

while((current_theta-previous_theta).all() >= tolerance):
    
    a=np.sum(current_theta[0]-alpha * (1/len(Y)) * np.sum((np.dot(X,current_theta)-Y)))
    b=np.sum(current_theta[1] - alpha * (1/len(Y)) * np.sum(np.dot(X_1,(np.dot(X,current_theta)-Y))))
    
    previous_theta=current_theta
    
    theta0.append(a)
    
    theta1.append(b)
    
    current_theta= np.array([[a],[b]])
    
    
    thetahistory.append(current_theta)
    
    cost_history.append( costfunction(X,Y,current_theta))
   
    
print(current_theta)    

t2=current_theta[0]+X_scaled*current_theta[1]
plt.title("Hypothesis") 
plt.xlabel("x axis ") 
plt.ylabel("y axis") 
plt.plot(X_scaled,t2)
plt.plot(X_scaled,y,'go')
plt.show()
    


# In[237]:


fig = plt.figure()
ax = plt.axes(projection="3d")

t0,t1= np.meshgrid(theta0,theta1)
cost_history=np.array(cost_history).reshape(-1,1)

ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('J(theta)')

ax.scatter3D(theta0,theta1,cost_history, cmap='hsv')

plt.show()


# In[238]:


ax = plt.axes(projection='3d')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('J(theta)')
ax.plot_wireframe(t0, t1,cost_history , color='orange')
ax.plot_surface(t0, t1, cost_history, rstride=1, cstride=1,cmap='winter', edgecolor='none')
ax.set_title('surface');


# In[251]:


cs = plt.contourf(cost_histo.reshape(334,2), levels=[10, 30, 50],
    colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
cs.cmap.set_over('red')
cs.cmap.set_under('blue')
cs.changed()


# In[250]:


print(theta0)
print(t0,t1)
print(cost_history.meshgrd())


# In[ ]:




