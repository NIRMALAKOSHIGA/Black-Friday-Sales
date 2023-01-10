#!/usr/bin/env python
# coding: utf-8

# # Black Friday Sales Prediction
# This dataset comprises of sales transactions captured at a retail store. It’s a classic dataset to explore and expand your feature engineering skills and day to day understanding from multiple shopping experiences. This is a regression problem. The dataset has 550,069 rows and 12 columns.
# 
# Problem: Predict purchase amount.
# 
# Data Overview
# Dataset has 537577 rows (transactions) and 12 columns (features) as described below:
# 
# User_ID: Unique ID of the user. There are a total of 5891 users in the dataset.
# Product_ID: Unique ID of the product. There are a total of 3623 products in the dataset.
# Gender: indicates the gender of the person making the transaction.
# Age: indicates the age group of the person making the transaction.
# Occupation: shows the occupation of the user, already labeled with numbers 0 to 20.
# City_Category: User's living city category. Cities are categorized into 3 different categories 'A', 'B' and 'C'.
# Stay_In_Current_City_Years: Indicates how long the users has lived in this city.
# Marital_Status: is 0 if the user is not married and 1 otherwise.
# Product_Category_1 to _3: Category of the product. All 3 are already labaled with numbers.
# Purchase: Purchase amount.

# # 1. import library

# In[8]:


# manipulation data
import pandas as pd
import numpy as np
import os
#visualiation data
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot

#default theme
plt.style.use('ggplot')
sns.set(context='notebook', style='darkgrid', palette='colorblind', font='sans-serif', font_scale=1, rc=None)
matplotlib.rcParams['figure.figsize'] =[8,8]
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = 'sans-serif'


# # 2. data analysis

# In[12]:


os.chdir('D:/projects/block friday sales')
train = pd.read_csv("train1.csv")
test = pd.read_csv('test1.csv')
train.head(5)


# In[13]:


train.shape


# like we c her we had 
# * 550068 rows 
# * 12 coluns

# In[14]:


train.info()


# In[17]:


train.dtypes.value_counts().plot.pie(explode=[0.01,0.01,0.01],autopct='%1.2f%%',shadow=True)
plt.title('type of our data');


# In[19]:


# show the numirical values

num_columns = [f for f in train.columns if train.dtypes[f] != 'object']
num_columns.remove('Purchase')
num_columns.remove('User_ID')
num_columns


# In[20]:


# show the categorical values

cat_columns = [f for f in train.columns if train.dtypes[f] == 'object']
cat_columns


# In[21]:


train.describe(include='all')


# A basic observation is that:
# 
# * Product P00265242 is the most popular product.
# * Most of the transactions were made by men.
# * Age group with most transactions was 26-35.
# * City Category with most transactions was B
# 
# but we will cover each of these in more depth later

# ## finding missing values

# In[22]:


missing_values=train.isnull().sum()
percent_missing = train.isnull().sum()/train.shape[0]*100

value = {
    'missing_values':missing_values,
    'percent_missing':percent_missing
}
frame=pd.DataFrame(value)
frame


# In[24]:


missing_values = train.isnull().sum()
missing_values = missing_values[missing_values > 0]
missing_values.sort_values(inplace=True)
missing_values.plot.pie(explode=[0.01,0.01],autopct='%1.1f%%',shadow=True)
plt.title('our missing values');


# Only Product_Category_2 and Product_Category_3 have null values which is good news. 
# However Product_Category_3 is null for nearly 70% of transactions so it can't give us much information.
# so we gonna drop Product_Category_3

# #### Product_Category_2

# In[25]:


train.Product_Category_2.value_counts()


# In[26]:


train.Product_Category_2.describe()


# In[28]:


# Replace using median 
median = train['Product_Category_2'].median()
train['Product_Category_2'].fillna(median, inplace=True)


# #### Product_Category_3

# In[29]:


train.Product_Category_3.value_counts()


# In[30]:


# drop Product_Category_3 
train=train.drop('Product_Category_3',axis=1)


# In[31]:


missing_values=train.isnull().sum()
percent_missing = train.isnull().sum()/train.shape[0]*100

value = {
    'missing_values':missing_values,
    'percent_missing':percent_missing
}
frame=pd.DataFrame(value)
frame


# # 3. data visualization

# In[32]:


train.hist(edgecolor='black',figsize=(12,12));


# In[33]:


train.columns


# ### A) Gender

# In[36]:


# pie chart 

size = train['Gender'].value_counts()
labels = ['Male', 'Female']
colors = ['#C4061D', 'green']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (5,5)
plt.pie(size, colors = colors, labels = labels, shadow = True, explode = explode, autopct = '%.2f%%')
plt.title('A Pie Chart representing the gender gap', fontsize = 10)
plt.axis('off')
plt.legend()
plt.show()


# In[37]:


sns.countplot(x=train.Gender)
plt.title('Gender per transaction');


# ### B) Age

# In[38]:


ageData = sorted(list(zip(train.Age.value_counts().index, train.Age.value_counts().values)))
age, productBuy = zip(*ageData)
age, productBuy = list(age), list(productBuy)
ageSeries = pd.Series((i for i in age))

data = [go.Bar(x=age, 
               y=productBuy, 
               name="How many products were sold",
               marker = dict(color=['black', 'yellow', 'green', 'blue', 'red', 'gray', '#C4061D'],
                            line = dict(color='#7C7C7C', width = .5)),
              text="Age: " + ageSeries)]
layout = go.Layout(title= "How many products were sold by ages")
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ### C) the occupation of customers

# In[40]:


palette=sns.color_palette("Set2")


# In[41]:



plt.rcParams['figure.figsize'] = (18, 9)
sns.countplot(train['Occupation'], palette = palette)
plt.title('Distribution of Occupation across customers', fontsize = 20)
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.show()


# #### *Total Money Spent per Occupation*

# In[42]:


spent_by_occ = train.groupby(by='Occupation').sum()['Purchase']
plt.figure(figsize=(20, 7))

sns.barplot(x=spent_by_occ.index,y=spent_by_occ.values)
plt.title('Total Money Spent per Occupation')
plt.show()


# Once again, the distribution of the mean amount spent within each occupation appears to mirror the distribution of the amount of people within each occupation. This is fortunate from a data science perspective, as we are not working with odd or outstanding features. Our data, in terms of age and occupation seems to simply make sense.

# ### d) City_Category

# In[45]:


plt.rcParams['figure.figsize'] = (9, 9)
sns.countplot(train['City_Category'], palette = palette)
plt.title('Distribution of Cities across customers', fontsize = 10)
plt.xlabel('Cities')
plt.ylabel('Count')
plt.show()


# ### E) Products

# Here we explore the products themselves. This is important, as we do not have labeled items in this dataset. Theoretically, a customer could be spending $5,000 on 4 new TVs, or 10,000 pens. This difference matters for stores, as their profits are affected. Since we do not know what the items are, let's explore the categories of the items.

# In[46]:


plt.figure(figsize=(10,6))
prod_by_cat = train.groupby('Product_Category_1')['Product_ID'].nunique()

sns.barplot(x=prod_by_cat.index,y=prod_by_cat.values, palette=palette)
plt.title('Number of Unique Items per Category')
plt.show()


# Category labels 1, 5, and 8 clearly have the most items within them. This could mean the store is known for that item, or that the category is a broad one.

# In[48]:


category = []
mean_purchase = []


for i in train['Product_Category_1'].unique():
    category.append(i)
category.sort()

for e in category:
    mean_purchase.append(train[train['Product_Category_1']==e]['Purchase'].mean())

plt.figure(figsize=(10,6))

sns.barplot(x=category,y=mean_purchase)
plt.title('Mean of the Purchases per Category')
plt.xlabel('Product Category')
plt.ylabel('Mean Purchase')
plt.show()


# In[49]:


# visualizing the different product categories

plt.rcParams['figure.figsize'] = (15, 25)
plt.style.use('ggplot')

plt.subplot(4, 1, 1)
sns.countplot(train['Product_Category_1'], palette = palette)
plt.title('Product Category 1', fontsize = 20)
plt.xlabel('Distribution of Product Category 1')
plt.ylabel('Count')

plt.subplot(4, 1, 2)
sns.countplot(train['Product_Category_2'], palette = palette)
plt.title('Product Category 2', fontsize = 20)
plt.xlabel('Distribution of Product Category 2')
plt.ylabel('Count')


plt.show()


# ## the purchase attribute which is our target variable

# In[50]:


# importing important libraries
from scipy import stats
from scipy.stats import norm


# In[51]:


# plotting a distribution plot for the target variable
plt.rcParams['figure.figsize'] = (20, 7)
sns.distplot(train['Purchase'], color = 'green', fit = norm)

# fitting the target variable to the normal curve 
mu, sigma = norm.fit(train['Purchase']) 
print("The mu {} and Sigma {} for the curve".format(mu, sigma))

plt.title('A distribution plot to represent the distribution of Purchase')
plt.legend(['Normal Distribution ($mu$: {}, $sigma$: {}'.format(mu, sigma)], loc = 'best')
plt.show()


# # data selection 

# first we gonna drop the :
# 1. User_ID	
# 2. Product_ID

# In[52]:


train = train.drop(['Product_ID','User_ID'],axis=1)


# In[53]:


# checking the new shape of data
print(train.shape)
train


# ## label encoding

# In[54]:


df_Gender = pd.get_dummies(train['Gender'])
df_Age = pd.get_dummies(train['Age'])
df_City_Category = pd.get_dummies(train['City_Category'])
df_Stay_In_Current_City_Years = pd.get_dummies(train['Stay_In_Current_City_Years'])

data_final= pd.concat([train, df_Gender, df_Age, df_City_Category, df_Stay_In_Current_City_Years], axis=1)

data_final.head()


# In[55]:


data_final = data_final.drop(['Gender','Age','City_Category','Stay_In_Current_City_Years'],axis=1)
data_final


# In[56]:


data_final.dtypes


# ### Predicting the Amount Spent
# 
# we will use one of the simplest machine learning models, i.e. the linear regression model, to predict the amount spent by the customer on Black Friday.
# 
# Linear regression represents a very simple method for supervised learning and it is an effective tool for predicting quantitative responses. You can find basic information about it right here: Linear Regression in Python
# 
# This model, like most of the supervised machine learning algorithms, makes a prediction based on the input features. The predicted output values are used for comparisons with desired outputs and an error is calculated. The error signal is propagated back through the model and model parameters are updating in a way to minimize the error. Finally, the model is considered to be fully trained if the error is small enough. This is a very basic explanation and we are going to analyze all these processes in details in future articles.

# ## split data

# In[57]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[58]:


x=data_final.drop('Purchase',axis=1)
y=data_final.Purchase


# In[59]:


print(x.shape)
print(y.shape)


# In[60]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)


# ### Feature Scaling

# In[61]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# ## 1) LinearRegression

# In[75]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train, y_train)
print(lm.fit(x_train, y_train))


# In[66]:


LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
         normalize=False)


# In[67]:


print('Intercept parameter:', lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_, x.columns, columns=['Coefficient'])
print(coeff_df)


# In[68]:


predictions = lm.predict(x_test)
print("Predicted purchases (in dollars) for new costumers:", predictions)


# In[70]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))


# In[77]:


lm.score(x_train,y_train )# R Square


# In[ ]:




