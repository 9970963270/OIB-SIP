#!/usr/bin/env python
# coding: utf-8

# # Oasis Infobytes : Data Science Internship

# ### Task 2 : To know what affects unemployement rate during Covid-19

# #### Intern name : Nilam Anil Ghadage

# In this task the data visualization is done to visualize the unemployment rate and also estimated correlation among the dataset variables and also visualized statewise unemployment rate.

# #### Step 1 : Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Loading dataset
data = pd.read_csv("C:\\Users\\Nilam\\Downloads\\Unemployment in India.csv")
data = pd.read_csv("C:\\Users\\Nilam\\Downloads\\Unemployment_Rate_upto_11_2020.csv")


# In[3]:


data.head(15)


# In[4]:


data.tail(15)


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data.describe()


# #### Step 2 : Exploratory Data Analysis

# In[8]:


sns.heatmap(data.corr(),annot= True)


# From above heatmap it is seen that their is high correlation between longitude and Estimated Unemployment Rate

# In[9]:


sns.pairplot(data, hue="Region.1");


# In[10]:


freq = data['Region.1'].value_counts()
freq


# In[11]:


freq.plot(kind='pie',startangle = 90)
plt.legend()
plt.show()


# - From above pie diagram the north region is highly considered region

# In[12]:


freq = data['Region.1'].value_counts()
freq


# In[13]:


sns.countplot(x='Region.1',data=data)
plt.xticks(rotation=45)
plt.ylabel('Estimated Unemployment Rate (%)')


# - From above bar diagram it is seen that the north region has the highest umemployment rate during Covid - 19

# ## To analyze the data statewise

# In[14]:


import plotly.express as px


# In[15]:


data.columns =["States","Date","Frequency","Estimated Unemployment Rate",
              "Estimated Employed","Estimated Labour Participation Rate",
              "Region","longitude","latitude"]


# In[16]:


print(data)


# In[17]:


unemployment_data = data[["States","Region","Estimated Unemployment Rate"]]
figure = px.sunburst(unemployment_data,path=["Region","States"],
                    values="Estimated Unemployment Rate",
                    width=600,height=600, color_continuous_scale="RdY1Gn",
                    title="Unemployment rate in India during Covid-19")
figure.show()


# # Thank You !

# In[ ]:




