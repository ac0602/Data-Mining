#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




