#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
from sklearn import datasets
from matplotlib import pyplot as plt


# In[4]:


cancer = datasets.load_breast_cancer() 
data=pd.DataFrame(cancer.data) ; data 
feature=pd.DataFrame(cancer.feature_names) ; feature 


data.columns=feature[0] 
target=pd.DataFrame(cancer.target)
target.columns=['target']

df=pd.concat([data,target], axis=1)
df.head()


# In[6]:


#2차원 공간에 가시화 시키기
df_f = df.copy()


# In[22]:


fig = plt.figure(figsize=(5,5)) 
X = df_f 
for col_n in range(1,4):
    print(col_n)
    plt.plot( X.iloc[:,0] , X.iloc[:,col_n] , 'o' , markersize=2 , color='blue' ) 
    plt.xlabel(data.columns[0] ) 
    plt.ylabel(data.columns[col_n] ) 
    plt.show()


# In[8]:


# 3차원 그리기 
fig = plt.figure(figsize=(8, 8)) 
ax = fig.add_subplot(111, projection='3d') 
X = df_f 
# 3d scatterplot 그리기 
print('X=',data.columns[0] ,'Y=', data.columns[1] , 'Z=',data.columns[3] )
ax.scatter( X.iloc[:,1] , X.iloc[:,2] , X.iloc[:,4] 
           ,s=10 #사이즈 
           ,cmap="orange" #컬러맵 
           ,alpha=1 #투명도 
          ) 

ax.set_xlabel(data.columns[0] )
ax.set_ylabel(data.columns[1] )
ax.set_zlabel(data.columns[3] )

plt.show()


# In[9]:


from sklearn.cluster import KMeans


ks = range(1,10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(df_f)
    inertias.append(model.inertia_)


plt.figure(figsize=(4, 4))

plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# In[23]:


clust_model = KMeans(n_clusters = 3                                     
                     , n_init=10 
                     , max_iter=50  
                     , random_state = 42) 

clust_model.fit(df_f) 


centers = clust_model.cluster_centers_ 
pred = clust_model.predict(df_f) 
print(pd.DataFrame(centers)) 
print(pred[:10])


# In[24]:


clust_df = df_f.copy() 
clust_df['clust'] = pred 
clust_df.head()


# In[25]:


import seaborn as sns


X = clust_df


sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,1], data=df_f, hue=clust_model.labels_, palette='coolwarm')
plt.scatter(centers[:,0], centers[:,1], c='black', alpha=0.8, s=150)
plt.show()


sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,2], data=df_f, hue=clust_model.labels_, palette='coolwarm')
plt.scatter(centers[:,0], centers[:,2], c='black', alpha=0.8, s=150)
plt.show()

sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,3], data=df_f, hue=clust_model.labels_, palette='coolwarm')
plt.scatter(centers[:,0], centers[:,3], c='black', alpha=0.8, s=150)
plt.show()

#2차원으로 시각화하기 


# In[26]:


# 3차원으로 시각화하기
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X = clust_df


ax.scatter( X.iloc[:,0] , X.iloc[:,1] , X.iloc[:,2] , c = X.clust , s = 10 , cmap = "rainbow" , alpha = 1 )


ax.scatter(centers[:,0],centers[:,1],centers[:,2] ,c='black', s=200)

ax.set_xlabel(data.columns[0] )
ax.set_ylabel(data.columns[1] )
ax.set_zlabel(data.columns[2] )

plt.show()


# In[27]:


# 3차원으로 시각화하기
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X = clust_df


ax.scatter( X.iloc[:,1] , X.iloc[:,2] , X.iloc[:,3] , c = X.clust , s = 10 , cmap = "rainbow" , alpha = 1 )


ax.scatter(centers[:,1],centers[:,2],centers[:,3] ,c='black', s=200)

ax.set_xlabel(data.columns[1] )
ax.set_ylabel(data.columns[2] )
ax.set_zlabel(data.columns[3] )

plt.show()


# In[28]:


cluster_mean= clust_df.groupby('clust').mean() 
cluster_mean


# In[ ]:




