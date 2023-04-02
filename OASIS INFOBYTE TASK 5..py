#!/usr/bin/env python
# coding: utf-8

# ### Oasis Infobyte : Data Science Internship
# ### Task 5 : Sales Prediction Using Python
# ### Name of Intern: Nilam Anil Ghadage
# ### Batch -March Phase 2 OIBSIP

# ### Loading Packages and Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ### Loading Dataset

# In[2]:


data=pd.read_csv("C:\\Users\\HP-PC\\Desktop\\Nilam\\Advertising.csv")


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.isnull().sum()


# In[9]:


data.shape


# ### Exploratory Data Analysis

# In[10]:


data.corr()


# In[11]:


sns.set()
sns.heatmap(data.corr(),annot = True)


# In[12]:


# pairplot
sns.pairplot(data,palette="hls")


# In[13]:


# Box Plot
sns.set(style= "darkgrid")
fig,axs1=plt.subplots(2,2,figsize=(15,15))
sns.boxplot(data=data,y="TV",ax=axs1[0,0],color='green')
sns.boxplot(data=data,y="Radio",ax=axs1[0,1],color='skyblue')
sns.boxplot(data=data,y="Newspaper",ax=axs1[1,0],color='orange')
sns.boxplot(data=data,y="Sales",ax=axs1[1,1],color='yellow')


# ### Data Preprocessing

# In[14]:


data=data.drop(columns=["Unnamed: 0"])


# In[15]:


data


# ### Splitting Feactures and Target

# In[16]:


x=data.drop(["Sales"],1)


# In[17]:


x


# In[18]:


x.head()


# In[19]:


y=data["Sales"]
y.head()


# ### Splitting data into Train and Test

# In[20]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[21]:


X_train


# In[22]:


y_train


# In[23]:


print(x.shape,X_train.shape,X_test.shape)


# ### Standarization

# In[24]:


x.describe()


# In[25]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[26]:


X_train_std=sc.fit_transform(X_train)
X_train_std


# In[27]:


X_test_std=sc.fit_transform(X_test)
X_test_std


# ### Model Building

# In[28]:


from sklearn.linear_model import LinearRegression
model= LinearRegression()


# In[29]:


model.fit(X_train_std,y_train)


# In[30]:


model.predict(X_test_std)


# In[31]:


y_pred_model=model.predict(X_test_std)


# In[32]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[33]:


r2_score(y_test,y_pred_model)


# In[34]:


mean_absolute_error(y_test,y_pred_model)


# In[35]:


mean_squared_error(y_test,y_pred_model)


# ### Thank You !!!
