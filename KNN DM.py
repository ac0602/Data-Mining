#!/usr/bin/env python
# coding: utf-8

# In[11]:


from sklearn import datasets
import pandas as pd


# In[12]:


cancer = datasets.load_breast_cancer()
cancer


# In[13]:


print(cancer.keys())


# In[16]:



cancer['target']


# In[17]:


X = cancer.data[:, :2]  
y = cancer.target  


# In[18]:


import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')


# In[19]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

KNN_MODEL = KNeighborsClassifier()
KNN_MODEL.fit(x_train, y_train)
prediction = KNN_MODEL.predict(x_test)


print("Training error       : {0:.3f}".format(KNN_MODEL.score(x_train, y_train)))
print("Testing error        : {0:.3f}".format((prediction==y_test).mean()))



print("cross_val_score      : {0:.3f}".format(cross_val_score(KNN_MODEL, X, y, cv=10).mean()))


# In[20]:


train_acc = []
test_acc = []

for n in range(1,15):
    KNN_MODEL = KNeighborsClassifier(n_neighbors=n)
    KNN_MODEL.fit(x_train, y_train)
    prediction = KNN_MODEL.predict(x_test)
    train_acc.append(KNN_MODEL.score(x_train, y_train))
    test_acc.append((prediction==y_test).mean())


# In[21]:




import numpy as np
plt.figure(figsize=(12, 9))
plt.plot(range(1, 15), train_acc, label='Training Dataset')
plt.plot(range(1, 15), test_acc, label='Test Dataset')
plt.xlabel("k")
plt.ylabel("accuracy")
plt.xticks(np.arange(0, 16, step=1))
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




