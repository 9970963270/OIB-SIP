#!/usr/bin/env python
# coding: utf-8

# # Oasis Infobytes : Data Science Internship

# ### Task 4 : To build an email spam detector machine learning model to detect the mail is spam or non-spam

# #### Intern name : Nilam Anil Ghadage
# #### March- P2 Batch Oasis Infobyte SIP

# #### Step 1 : Data loading

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


raw_data = pd.read_csv("spam.csv", encoding="latin-1")
raw_data


# #### Step 2 : Data Cleaning

# In[3]:


raw_data.columns


# In[4]:


raw_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)


# In[5]:


raw_data.head(30)


# In[6]:


raw_data.rename(columns={'v1':'target','v2':'mails'},inplace =True)
raw_data


# In[7]:


raw_data.isnull().sum()


# In[8]:


raw_data.info()


# In[9]:


raw_data.shape


# In[10]:


# To Check duplicate values in dataset
raw_data.duplicated().sum()


# In[11]:


raw_data =raw_data.drop_duplicates(keep='first')


# In[12]:


raw_data.duplicated().sum()


# #### Step 3 : Exploratory data analysis

# In[13]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[14]:


raw_data['target'] = encoder.fit_transform(raw_data['target'])


# In[15]:


raw_data.head()


# In[16]:


raw_data['target'].value_counts()


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


plt.pie(raw_data['target'].value_counts(), labels = ['ham','spam'],autopct="%0.2f")
plt.show()


# - From above pie chart it is seen that 87.37% email are ham thus 12.63% mail are spam

# #### Step 4 : Performing Navie Bayesian

# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB


# In[20]:


X = raw_data['target']
y = raw_data['mails']


# In[21]:


print(X)


# In[22]:


X_train,X_test,y_train,y_test=train_test_split(raw_data.target, raw_data.mails , test_size = 0.25)


# In[23]:


cv = CountVectorizer()
y_train_count = cv.fit_transform(y_train.values)


# In[24]:


y_train_count.toarray()


# In[25]:


# To analyze the model
model = MultinomialNB()
model.fit(y_train_count, X_train)


# In[26]:


# To test it is ham mail or not
mail_ham = ["I HAVE A DATE ON SUNDAY WITH WILL!!"]
mail_ham_count = cv.transform(mail_ham)
model.predict(mail_ham_count)


# - The array is 0 that is the email is not spam hence it is ham mail

# In[27]:


# To test it is spam mail or not
mail_spam = ["Did you hear about the new \Divorce Barbie\"? It comes with all of Ken's stuff!"]
mail_spam_count = cv.transform(mail_spam)
model.predict(mail_spam_count)


# - The array is 0 that is the email is not ham hence it is spam mail

# In[28]:


y_test_count = cv.transform(y_test)
model.score(y_test_count, X_test)


# - Hence the model has 99 % accuracy thus model is good

# #### Step 5 : Performing Logistic Regression

# In[29]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[30]:


X = raw_data['mails']
y = raw_data['target']


# In[31]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2, random_state = 3)


# In[32]:


feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase ='True')


# In[33]:


print(X)


# In[34]:


y_train = y_train.astype('int')
y_test = y_test.astype('int')


# In[35]:


X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


# In[36]:


print(X_train)


# In[37]:


print(X_train_features)


# In[38]:


# Training the Ml Logistic regression model
model = LogisticRegression()


# In[39]:


model.fit(X_train_features, y_train)


# In[40]:


predict_train_data = model.predict(X_train_features)
accuracy_train_data = accuracy_score(y_train, predict_train_data)


# In[41]:


print('accuracy_train_data:', accuracy_train_data)


# - The accuracy for training data is 96%

# In[42]:


predict_test_data = model.predict(X_test_features)
accuracy_test_data = accuracy_score(y_test, predict_test_data)


# In[43]:


print('accuracy_test_data:', accuracy_test_data)


# - The accuracy for test data is 96%

# ## Thank You !

# In[ ]:




