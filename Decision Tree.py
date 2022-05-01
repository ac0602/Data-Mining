#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# In[6]:


cancer=load_breast_cancer()


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=24)


# In[8]:


from sklearn.tree import DecisionTreeClassifier   



DT_MODEL= DecisionTreeClassifier(random_state=0) 
DT_MODEL.fit(X_train, y_train)



prediction=DT_MODEL.predict(X_test)


# In[9]:




from sklearn.metrics import classification_report, confusion_matrix
CM=confusion_matrix(y_test, prediction)
CM_report=classification_report(y_test, prediction)
print('-'*15, 'Confusion Matrix','-'*15)
print(CM)

print('-'*15, 'Confusion Matrix2','-'*15)
import pandas as pd
CM_rename=pd.DataFrame(CM).rename(index={0:'실제값(0)',1:'실제값(1)',2:'실제값(2)'},columns={0:'예측값(0)',1:'예측값(1)',2:'예측값(2)'})
print(CM_rename)


# In[10]:




print('-'*20, '성능평가','-'*20)
print(CM_report)


# In[12]:




import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(15,10))
plot_tree(DT_MODEL,feature_names=cancer.feature_names, class_names=cancer.target_names, filled=True, fontsize=9)
plt.show()


# In[14]:




DT_MODEL_DEP3= DecisionTreeClassifier(max_depth=3, random_state=0) 
DT_MODEL_DEP3.fit(X_train, y_train)

prediction_DEP3=DT_MODEL_DEP3.predict(X_test)

CM_DEP3=confusion_matrix(y_test, prediction_DEP3)
CM_report_DEP3=classification_report(y_test, prediction_DEP3)

print('-'*15, 'Confusion Matrix','-'*15)
print(CM_DEP3)
print('-'*20, '성능평가','-'*20)
print(CM_report_DEP3)

plt.figure(figsize=(15,10))
plot_tree(DT_MODEL_DEP3,feature_names=cancer.feature_names, class_names=cancer.target_names, filled=True, fontsize=9)
plt.show()


# In[20]:


# 가지치기2 - 최소 샘플 개수 설정

DT_MODEL_MINSAM= DecisionTreeClassifier(min_samples_split=40, random_state=0) 
DT_MODEL_MINSAM.fit(X_train, y_train)

prediction_MINSAM=DT_MODEL_MINSAM.predict(X_test)

CM_MINSAM=confusion_matrix(y_test, prediction_MINSAM)
CM_report_MINSAM=classification_report(y_test, prediction_MINSAM)

print('-'*15, 'Confusion Matrix','-'*15)
print(CM_MINSAM)
print('-'*20, '성능평가','-'*20)
print(CM_report_MINSAM)

plt.figure(figsize=(15,10))
plot_tree(DT_MODEL_MINSAM,feature_names=cancer.feature_names, class_names=cancer.target_names, filled=True, fontsize=9)
plt.show()


# In[22]:


# 변수 중요도

for name, value in zip(cancer.feature_names , DT_MODEL_DEP3.feature_importances_):
    print('{0} : {1:.3f}'.format(name, value))
    
# 변수 중요도 가시화    

import seaborn as sns
sns.barplot(x=DT_MODEL_DEP3.feature_importances_ , y=cancer.feature_names)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




