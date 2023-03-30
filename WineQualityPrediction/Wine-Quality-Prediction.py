#!/usr/bin/env python
# coding: utf-8

# **IMPORT MODULES**

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# **LOADING THE DATASET**

# In[17]:


dataset = pd.read_csv('winequalityN.csv')
dataset.head()


# In[7]:


#STATISTICAL INFORMATION
dataset.describe()


# In[18]:


#DATATYPE INFORMATION
dataset.info()


# **PREPROCESSING THE DATASET**

# In[19]:


#CHECKING FOR NULL VALUES
dataset.isnull().sum()


# In[20]:


#FILLINF UP THE MISSING VALUES WITH A FOR LOOP
for col, value in df.items():
    #WE WILL IGNORE TYPE COL BECAUSE IT IS A STRING
    if col != 'type':
       dataset[col] = dataset[col].fillna(dataset[col].mean())
    #WE USE MEAN() O FILL THE MEAN VALUES OF THAT SPECIFIC ATTRIBUTE


# In[21]:


dataset.isnull().sum()


# **EXPLORATORY DATA ANALYSIS**

# In[22]:


#CREATING BOX PLOT OF THE ATTRIBUTES FOR US TO CHECK THE OUTLINERS
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    if col != 'type':
        sns.boxplot(y=col, data=df, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[23]:


#CREATING DIST PLOT FOR US TO EXPLORE THE DISTRIBUTION PLOT OF ALL NUMERICAL ATTRIBUTES
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    if col != 'type':
        sns.distplot(value, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[26]:


#LOG TRANSFORMATION
dataset['free sulfur dioxide'] = np.log(1 + dataset['free sulfur dioxide'])
#RIGHT BELOW WE CAN SEE THERE IS A BELL CURVE


# In[25]:


sns.distplot(dataset['free sulfur dioxide'])


# In[28]:


sns.countplot(dataset['type'])
#THE CAPACITY OF EACH WINE TYPE


# In[29]:


sns.countplot(dataset['quality'])
#IT IS VISIBLE THAT THE MIDDLE CLASSES HAVE HIGHER COUNTS


# **CORRELATION MATRIX**

# In[34]:


#THIS CORRELATION MATRIX TABLE SHOWS US THE CORRELATION BETWEEN TWO VARIABLES
corr = dataset.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')


# **INPUT SPLIT**

# In[35]:


#SPLITTING THE DATASET
X = dataset.drop(columns=['type', 'quality'])
y = dataset['quality']


# **CLASS IMBALACEMENT**

# In[36]:


y.value_counts()


# In[40]:


#ABOVE WE SEE THE COUNT OF DATA FOR EACH CLASS
#WITH SMOTE WE ARE TRYING TO BALANCE THE CLASS RATIO
from imblearn.over_sampling import SMOTE
oversample = SMOTE(k_neighbors=4)
# transform the dataset
X, y = oversample.fit_resample(X, y)


# In[42]:


y.value_counts()
#BELOW THE CLASSES HAVE THE SAME UPPERVALUE


# **MODEL TRAINING**

# In[43]:


#FUNCTION
from sklearn.model_selection import cross_val_score, train_test_split
def classify(model, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    #HERE WE ARE TRAINING THE MODEL
    model.fit(x_train, y_train)
    print("Accuracy:", model.score(x_test, y_test) * 100)
    
    # CROSSING VALIDATION
    score = cross_val_score(model, X, y, cv=5)
    print("CV Score:", np.mean(score)*100)


# In[44]:


#THIS IS NOT A REGRESSION MODEL IT IS A CLASSIFIATION ONE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model, X, y)


# In[45]:


#WE USE DICISION TREE
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model, X, y)


# In[46]:


#WE USE RANDOME FOREST
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify(model, X, y)


# In[47]:


#WE USE SOME EXTRA TREES
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
classify(model, X, y)


# In[ ]:





# In[ ]:




