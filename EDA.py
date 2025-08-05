#!/usr/bin/env python
# coding: utf-8

# # Univariate Analysis

# In[3]:


import pandas as pd


# In[5]:


import seaborn as sns


# In[6]:


df =pd.read_csv("train.csv")


# In[5]:


df.head()


# # 1.Categorical data

# ## Countplot

# In[33]:


sns.countplot(df['Survived'])
df["Survived"].value_counts().plot(kind='bar')


# ## Piechart

# In[50]:


df['Survived'].value_counts().plot(kind='pie',autopct='%.2f')


# # 2.Numerical

# ## Histogram

# In[62]:


import matplotlib.pyplot as plt


# In[64]:


plt.hist(df['Age'])


# ## Distplot

# In[67]:


sns.distplot(df['Age']) # PDA = probabbility density function


# ## Boxplot

# In[72]:


sns.boxplot(df['Fare'])


# In[74]:


df.describe()


# In[76]:


df['Age'].min()


# In[78]:


df['Age'].max()


# In[80]:


df['Age'].skew()


# In[11]:


df.isnull().sum()


# In[15]:


df.duplicated()


# In[27]:


df.head(10)


# # Multivariate or Bivariate analysis

# In[41]:


import pandas as pd


# In[43]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[19]:


titanic = pd.read_csv('train.csv')


# In[62]:


flights = sns.load_dataset('flights')


# In[64]:


tips = sns.load_dataset('tips')


# In[50]:


iris = sns.load_dataset('iris')


# ## 1.Scatterplot (Numerical-Numerical)

# In[71]:


tips.head()


# In[87]:


# Bi-varaite Analysis
sns.scatterplot(x='total_bill', y='tip', data=tips)
# Multivariate Analysis
sns.scatterplot(x='total_bill', y='tip', data=tips,hue='sex',style='smoker',size='size')


# ## 2. Bar plot (Numerical-categorical)

# In[93]:


sns.barplot(x='Pclass',y='Age',data=titanic)
#Multivariate Analysis
sns.barplot(x='Pclass',y='Age',data=titanic,hue='Sex')


# ## 3.Box plot

# In[98]:


sns.boxplot(x='Sex',y='Age',data=titanic)


# ## 4.Dist Plot(Numerical-categorical)

# In[27]:


sns.distplot(titanic[titanic['Survived']==0]['Age'],hist=False)
sns.distplot(titanic[titanic['Survived']==1]['Age'],hist=False)


# ## 5.Heatmap (categorical-categorical)

# In[34]:


sns.heatmap(pd.crosstab(titanic['Pclass'],titanic['Survived']))


# ## 6.ClusterMap (categorical-categorical)

# In[47]:


sns.clustermap(pd.crosstab(titanic['Parch'],titanic['Survived']))


# ## 7.Pairplot (numerical-numerical)

# In[52]:


iris.head()


# In[59]:


sns.pairplot(iris,hue='species')


# ## 8.Lineplot(Numerical-Numerical)

# In[66]:


flights.head()


# In[78]:


new=flights.groupby('year').sum(numeric_only=True).reset_index()


# In[80]:


new


# In[84]:


sns.lineplot(x=new['year'],y=new['passengers'])


# In[86]:


flights.pivot_table(values='passengers',index='month',columns='year')


# In[ ]:




