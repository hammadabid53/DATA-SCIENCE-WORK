#!/usr/bin/env python
# coding: utf-8

# In[269]:


#NAME : HAMMAD ABID ,  ID:9134

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[270]:


#Reading csv_file
df = pd.read_csv("carprices.csv")


# In[271]:


df.head()


# In[272]:


#converting string data to numeric using one hot Encoding
carmodel = pd.get_dummies(df[['Car Model']], drop_first= True)
carmodel.head()


# In[273]:


#Now merging 
data = pd.concat([df, carmodel ], axis = 1)
data.head()


# In[274]:


categorical_features = ['Car Model']


# In[275]:


data = data.drop(columns=categorical_features, axis=1)
data.head()


# In[276]:


Y = data['Sell Price']
X = data.drop('Sell Price', axis=1)


# In[277]:


from sklearn.model_selection import train_test_split

np.random.seed(0)
trainX, testX, trainY, testY = train_test_split(X,Y, test_size = 0.3, random_state = (12))


# In[278]:


#LinearRegression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(trainX, trainY)


# In[279]:


Lm_predict = lm.predict(testX)
print("Prediction Using Linear Regression for test set: {}".format(Lm_predict))


# In[280]:


data_diff = pd.DataFrame({'Actual value': testY, 'Predicted value': Lm_predict})
data_diff.head()


# In[281]:


#Performance Measurement of Linear Regression
from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(testY, Lm_predict)
meanSqErr = metrics.mean_squared_error(testY, Lm_predict)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(testY, Lm_predict))
print('R squared: {:.2f}'.format(lm.score(X,Y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)


# In[282]:


#Implementing Decision Tree
from sklearn.tree import DecisionTreeClassifier
decision = DecisionTreeClassifier(random_state=1)
decision.fit(trainX, trainY)


# In[283]:


descision_predict = decision.predict(testX)
print("Prediction Using Decision for test set: {}".format(descision_predict))


# In[284]:


data_diff = pd.DataFrame({'Actual value': testY, 'Predicted value':descision_predict})
data_diff.head()


# In[309]:


#Performance Measurement of Decision
from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(testY,descision_predict)
meanSqErr = metrics.mean_squared_error(testY, descision_predict)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(testY, descision_predict))
print('R squared: {:.2f}'.format(decision.score(X,Y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)


# In[286]:


# Implementing KNN
from sklearn.neighbors import KNeighborsClassifier
kc = KNeighborsClassifier()
kc.fit(trainX, trainY)


# In[287]:


KN_predict = kc.predict(testX)
print("Prediction Using KNN for test set: {}".format(KN_predict))


# In[288]:


data_diff = pd.DataFrame({'Actual value': testY, 'Predicted value':KN_predict})
data_diff.head()


# In[310]:


#Performance Measurement of KNN
from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(testY,KN_predict)
meanSqErr = metrics.mean_squared_error(testY, KN_predict)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(testY, KN_predict))
print('R squared: {:.2f}'.format(kc.score(X,Y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)


# In[319]:


#now calculating the accuracy 
print(f'Linear Model Test Accuracy: {lm.score(trainX, trainY)}')
print(f'Decision Model Test Accuracy: {decision.score(trainX, trainY)}')
print(f'KNN Model Test Accuracy: {kc.score(trainX, trainY)}')


# In[314]:


print(f'Linear Model Accuracy: {lm.score(testX,Lm_predict)*100}%')
print(f'Decision Tree Model Accuracy: {decision.score(testX, descision_predict)*100}%')
print(f'KNN Model Accuracy: {kc.score(testX,KN_predict)*100}%')


# In[ ]:


QUESTION1 part NO3

#as above data shows that :
#linear model:best fit we can accept this
#Decision model:best fit we can accept this
#KNN model:overifit


# In[ ]:




