#!/usr/bin/env python
# coding: utf-8

# # Oasis Infobytes : Data Science Internship

# ### Task 1 : Train a ML model for iris dataset

# #### Intern name : Nilam Anil Ghadage

# In this task the ML model is built for classification of iris datset.

# #### Step 1 : Importing Libraries

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# #### Step 2 : Importing Dataset

# In[2]:


data = pd.read_csv("C:\\Users\\Nilam\\Downloads\\IRIS DATA.csv")
data


# In[3]:


data.head() # For first 5 rows


# In[4]:


data.tail() # For last 5 rows


# In[5]:


data.shape #Dimension of dataset


# In[6]:


data.describe() # descriptive Statistics


# In[7]:


data.info() # Information about dataset


# The iris dataset have one target column which is "Species" it is categorical. It has 4 variables of iris and none of the cells have null values.

# #### Step 2 : Exploratory Data Analysis

# In[8]:


# Using heatmap to know the correlation among the variables


# In[9]:


data1 = data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]]
print(data1.head())


# In[10]:


sns.heatmap(data1.corr(),annot= True)


# From above heatmap it is seen that petal length and petal width are highly correlated.

# In[11]:


sns.boxplot(x=data.Species,y=data.SepalLengthCm)
plt.show()


# From above boxplot in "Iris-virginica" one outlier is detected.

# In[12]:


sns.boxplot(x=data.Species,y=data.PetalLengthCm)
plt.show()


# From above boxplot in "Iris-versicolor" one outlier is detected.

# #### Step 3 : Building ML model for classification

# In[13]:


from sklearn.linear_model import LogisticRegression


# In[14]:


# Dividing dataset into train and test data
X=data1.drop('Species',axis = 1)
X


# In[15]:


y=data1['Species']
y


# In[16]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30,random_state=21)


# In[17]:


model = LogisticRegression()


# In[18]:


model.fit(X_train,y_train)


# In[19]:


model_pred= model.predict(X_test)


# In[20]:


from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


# In[21]:


print("Accuracy:", accuracy_score(model_pred,y_test)*100)


# In[22]:


confusion_matrix(model_pred,y_test)


# In[23]:


print(classification_report(model_pred,y_test))


# Interpretation : The above Logistic model has 93.33 accuracy which is quite enough that our model is good and therefore from confusion matrix it is seen that there are 3 misclassified variables in data.

# # THANK YOU

# In[ ]:




