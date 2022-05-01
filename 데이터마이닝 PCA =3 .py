#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import datasets
load_df=datasets.load_breast_cancer()

data=pd.DataFrame(load_df.data)
feature=pd.DataFrame(load_df.feature_names)
data.columns=feature[0]
target=pd.DataFrame(load_df.target)
target.columns=['target']
df=pd.concat([data,target], axis=1)
print(df.shape)
df.head()


# In[2]:


# dictionary 키값 확인
print(df.keys())


# In[3]:


#변수가 너무 많기에, 차원축소를 진행 해주었습니다
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# In[4]:


# 데이터 가져오기

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X = cancer.data


# In[5]:


pd.DataFrame(X)


# In[6]:


# Data Normalizing (Mean Centering)

from sklearn.preprocessing import StandardScaler
X_ = StandardScaler().fit_transform(X)


# In[17]:


# PCA 수행 - 주성분 3개
pca = PCA(n_components=3)
pc = pca.fit_transform(X_)


# In[18]:


# 카테고리정보 (음성 양성)
pc_y = np.c_[pc,y]
df = pd.DataFrame(pc_y,columns=['PC1','PC2','PC3','diagnosis'])


# In[19]:


#Plotting

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['PC1'], df['PC2'], df['PC3'], c=df['diagnosis'],  s=20)


# In[24]:


#Plotting

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['PC1'], df['PC2'], df['PC3'], c=df['diagnosis'],  s=20)


# In[30]:


# PCA 수행 - 주성분 3개
pca = PCA(n_components=3)
pc = pca.fit_transform(X_)

df_var = pd.DataFrame({'var':pca.explained_variance_ratio_,
             'PC':['PC1','PC2', 'PC3']})
sns.barplot(x='PC',y="var", 
           data=df_var, color="c");


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




