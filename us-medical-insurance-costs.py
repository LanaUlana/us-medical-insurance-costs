#!/usr/bin/env python
# coding: utf-8

# # U.S. Medical Insurance Costs

# ### Project Scope and research questions
# 
# * average age of the patients in the dataset
# * average bmi
# * where a majority of the individuals are from
# * different costs between smokers vs. non-smokers
# * what the average age is for someone who has at least one child in this dataset
# * features that are the most influential for an individual’s medical insurance charges based on analysis
# * build regression model
# 
# 
# 
# We will explore how a set of particular factors (age, sex, smoking, region of the US) influence insurance costs and build regression model based on available data. 

# In[1]:


import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('insurance.csv')
df.head()


# There are seven columns, some columns are numerical while some are categorical. Let's check if there is no missing data.

# In[3]:


df.isnull().sum()


# Great! Let's than get main stats for all the columns.
# 
# 

# In[4]:


df.describe()


# #### Inferences based on summary statistics
# 
# * age 
# 
# Age range is not that big, we have data for adults only, with no data on elder or children. Exclusion of elder adults may bias our dataset. 
# 
# * bmi
# 
# We notice that both mean and bmi within IQR are highher that what consider healthy bmi.
# 
# * children 
# 
# We need to explore if having children correlates with greater insurance cost, as children may be covered by parents health insurance.
# 
# * charges 
# 
# We observe that mean is higher than median, that indicates right skewness.
# 
# 
# 
# 
# 
# 

# To bettter get a grasp of a data we would visualize it with graphs.
# First, let's look at the distribution of cost. 

# In[5]:


plt.figure(figsize=(14,5))
plt.hist(df.charges, bins=20)
plt.xlabel('Insurance Cost, $')
plt.title('Insurance Cost Distribution')
plt.show()


# As was noticed, we have right-skewed distribution of charges with some outliers with extremely expensive health insurance.

# In[6]:


sns.countplot(data=df, x='region')
plt.show()


# In[7]:


sns.kdeplot(df[(df.region=='southwest')]["charges"], shade=True)
sns.kdeplot(df[(df.region=='southeast')]["charges"], shade=True)
sns.kdeplot(df[(df.region=='northwest')]["charges"], shade=True)
sns.kdeplot(df[(df.region=='northeast')]["charges"], shade=True)
plt.show()


# In[8]:


sns.countplot(data=df, x='sex')
plt.show()


# In[9]:


sns.kdeplot(df[(df.sex=='male')]["charges"], shade=True)
sns.kdeplot(df[(df.sex=='female')]["charges"], shade=True)
plt.show()


# In[10]:


plt.figure(figsize=(14,5))
plt.title("Distribution of age")
ax = sns.histplot(df["age"], bins=20, color = 'g')


# In[11]:


plt.figure(figsize=(14,5))
ax = plt.scatter(df.age, df.charges)
plt.show()


# So we've got even representation of both males/females and all four regions, however our dataset contains more datapoints on young adults. That may explain cost distribution we saw earlier and we could notice slight correlation between age and actual charges.
# 
# Lets look into costs of smokers vs non-smokers.

# In[12]:


plt.figure(figsize=(14,5))
plt.hist(df[(df.smoker == 'yes')]["charges"], bins=20, alpha=0.5, color='red')
plt.hist(df[(df.smoker == 'no')]["charges"], bins=20, alpha=0.5, color='green')
plt.show()


# Smoking patients spend more on insurance. But it looks like the number of non-smoking patients is greater. Lets's check this.
# 

# In[13]:


sns.catplot(x="smoker", kind="count",hue = 'sex', palette="rocket", data=df)
plt.show()


# There are significantly more non-smokers than smokers and we could see that there is more male smokers. Let's see if cost for them are higher.

# In[14]:


sns.catplot(x="sex", y="charges", hue="smoker", kind="violin", data=df, palette = 'husl')
plt.show()


# It's obviously better for your finances not to smoke, because not only you pay for cigarettes, but also your insurance would be more expensive. 
# 
# What about bmi, are there such strong relations between bmi and cost too? Let's look into it.

# In[15]:


plt.figure(figsize=(14,5))
plt.title("Distribution of bmi")
ax = sns.histplot(df["bmi"], color = 'r')


# In[16]:


plt.figure(figsize=(14,5))
ax = plt.scatter(df.bmi, df.charges)
plt.show()


# Quite intresting: bmi follows almost normal distribution, but it seems there is no obvious relationship between bmi and insurance cost.
# 
# Let us outline smokers on this scatterplot.
# 

# In[17]:


plt.figure(figsize=(14,5))
ax = sns.scatterplot(x='bmi',y='charges',data=df, hue='smoker')

plt.show()


# Now we see that smoking has the biggesr effect on cost. Let's see if number of children influences cost.

# In[18]:


plt.figure(figsize=(14,5))
plt.hist(df[(df.children > 0)]["charges"], bins=25, alpha=0.5, color='blue')
plt.hist(df[(df.children == 0)]["charges"], bins=25, alpha=0.5, color='yellow')
plt.show()


# It seems that children don't influence insurance cost that much.
# 
# Now, that we've gotten feel of our dataset let's prepare data for creting regression.
# Let's begin with encoding categorical features.
# 

# In[19]:


from sklearn.preprocessing import LabelEncoder
#sex
le = LabelEncoder()
le.fit(df.sex.drop_duplicates()) 
df.sex = le.transform(df.sex)
# smoker or not
le.fit(df.smoker.drop_duplicates()) 
df.smoker = le.transform(df.smoker)
#region
le.fit(df.region.drop_duplicates()) 
df.region = le.transform(df.region)


# In[20]:


df.corr()['charges'].sort_values()


# In[21]:


f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),
            square=True, ax=ax)
plt.show()


# A strong correlation is observed only with smoking. 
# 
# Let's predict insurance cost using diffrent regressions.

# In[22]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[23]:


scaler = StandardScaler()
x = scaler.fit_transform(df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']])
y = scaler.fit_transform(df[['charges']])


# In[24]:


x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)
mlr = LinearRegression().fit(x_train,y_train)

y_train_pred = mlr.predict(x_train)
y_test_pred = mlr.predict(x_test)

print(mlr.score(x_test,y_test))


# Let's see if we can do better with other algorithms.

# In[25]:


regressor = DecisionTreeRegressor(max_depth = 4)
regressor.fit(x_train, y_train)
predictions = regressor.predict(x_test)
print(regressor.score(x_test, y_test))


# A lit bit of tweaking of the max_depth parameter and we got ourselves 89% score.
# Not ideal, let's try random forest.

# In[26]:


tree = RandomForestRegressor(n_estimators = 32, random_state = 0, max_depth=4, n_jobs=-1)
tree.fit(x_train, y_train.ravel())
print(tree.score(x_test, y_test))
print(tree.score(x_train, y_train))
print(tree.feature_importances_)


# Through tuning max-depth and n_estimators only we found the best score for our random forest on this dataset. 
# 
# ### Conclusion
# Accuracy is not ideal, but we trained our algorithm to predict with roughly 90% accuracy the insurance cost, based on such factors as age, sex, bmi, children, smoker, region. 
# 
# We take into consideration that based on our data we heavily rely in our predictions on the fact of smoking and our data could be biased as we don't have data for elder patients. 
