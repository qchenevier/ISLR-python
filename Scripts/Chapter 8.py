
# coding: utf-8

# - [8.1.1 Regression Trees](#8.1.1-Regression-Trees)
# - [8.1.2 Classification Trees](#8.1.2-Classification-Trees)
# - [Lab: 8.3.1 Fitting Classification Trees](#8.3.1-Fitting-Classification-Trees)
# - [Lab: 8.3.2 Fitting Regression Trees](#8.3.2-Fitting-Regression-Trees)
# - [Lab: 8.3.3 Bagging and Random Forests](#8.3.3-Bagging-and-Random-Forests)
# - [Lab: 8.3.4 Boosting](#8.3.4-Boosting)

# # Chapter 8 - Tree-based Methods

# In[1]:

# %load ../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pydot
from IPython.display import Image

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.externals.six import StringIO  
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

pd.set_option('display.notebook_repr_html', False)

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-white')


# In[2]:

# This function creates images of tree models using pydot
def print_tree(estimator, features, class_names=None, filled=True):
    tree = estimator
    names = features
    color = filled
    classn = class_names
    
    dot_data = StringIO()
    export_graphviz(estimator, out_file=dot_data, feature_names=features, class_names=classn, filled=filled)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    return(graph)


# ### 8.1.1 Regression Trees

# In R, I exported the dataset from package 'ISLR' to a csv file.

# In[3]:

df = pd.read_csv('Data/Hitters.csv').dropna()
df.info()


# In[4]:

X = df[['Years', 'Hits']].as_matrix()
y = np.log(df.Salary.as_matrix())

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,4))
ax1.hist(df.Salary.as_matrix())
ax1.set_xlabel('Salary')
ax2.hist(y)
ax2.set_xlabel('Log(Salary)');


# In[5]:

regr = DecisionTreeRegressor(max_leaf_nodes=3)
regr.fit(X, y)


# ### Figure 8.1

# In[6]:

graph = print_tree(regr, features=['Years', 'Hits'])
Image(graph.create_png())


# ###  Figure 8.2

# In[7]:

df.plot('Years', 'Hits', kind='scatter', color='orange', figsize=(7,6))
plt.xlim(0,25)
plt.ylim(ymin=-5)
plt.xticks([1, 4.5, 24])
plt.yticks([1, 117.5, 238])
plt.vlines(4.5, ymin=-5, ymax=250)
plt.hlines(117.5, xmin=4.5, xmax=25)
plt.annotate('R1', xy=(2,117.5), fontsize='xx-large')
plt.annotate('R2', xy=(11,60), fontsize='xx-large')
plt.annotate('R3', xy=(11,170), fontsize='xx-large');


# ### Pruning
# This is currently not supported in scikit-learn. See first point under 'disadvantages of decision trees in the <A href='http://scikit-learn.github.io/dev/modules/tree.html#'>documentation</A>. Implementation has been <A href='https://github.com/scikit-learn/scikit-learn/pull/941'>discussed</A> but Random Forests have better predictive qualities than a single pruned tree anyway if I understand correctly.
#     

# ### 8.1.2 Classification Trees

# Dataset available on http://www-bcf.usc.edu/~gareth/ISL/data.html

# In[8]:

df2 = pd.read_csv('Data/Heart.csv').drop('Unnamed: 0', axis=1).dropna()
df2.info()


# In[9]:

df2.ChestPain = pd.factorize(df2.ChestPain)[0]
df2.Thal = pd.factorize(df2.Thal)[0]


# In[10]:

X2 = df2.drop('AHD', axis=1)
y2 = pd.factorize(df2.AHD)[0]


# In[11]:

clf = DecisionTreeClassifier(max_depth=None, max_leaf_nodes=6, max_features=3)
clf.fit(X2,y2)


# In[12]:

clf.score(X2,y2)


# In[13]:

graph2 = print_tree(clf, features=X2.columns, class_names=['No', 'Yes'])
Image(graph2.create_png())


# ## Lab

# ### 8.3.1 Fitting Classification Trees

# In R, I exported the dataset from package 'ISLR' to a csv file.

# In[14]:

df3 = pd.read_csv('Data/Carseats.csv').drop('Unnamed: 0', axis=1)
df3.head()


# In[15]:

df3['High'] = df3.Sales.map(lambda x: 1 if x>8 else 0)
df3.ShelveLoc = pd.factorize(df3.ShelveLoc)[0]

df3.Urban = df3.Urban.map({'No':0, 'Yes':1})
df3.US = df3.US.map({'No':0, 'Yes':1})
df3.info()


# In[16]:

df3.head(5)


# In[17]:

X = df3.drop(['Sales', 'High'], axis=1)
y = df3.High

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)


# In[18]:

clf = DecisionTreeClassifier(max_depth=6)
clf.fit(X, y)


# In[19]:

print(classification_report(y, clf.predict(X)))


# In[20]:

graph3 = print_tree(clf, features=X.columns, class_names=['No', 'Yes'])
Image(graph3.create_png())


# In[21]:

clf.fit(X_train, y_train)
pred = clf.predict(X_test)


# In[22]:

cm = pd.DataFrame(confusion_matrix(y_test, pred).T, index=['No', 'Yes'], columns=['No', 'Yes'])
cm.index.name = 'Predicted'
cm.columns.name = 'True'
cm


# In[23]:

# Precision of the model using test data is 74%
print(classification_report(y_test, pred))


# Pruning not implemented in scikit-learn.

# ### 8.3.2 Fitting Regression Trees

# In R, I exported the dataset from package 'MASS' to a csv file.

# In[24]:

boston_df = pd.read_csv('Data/Boston.csv')
boston_df.info()


# In[25]:

X = boston_df.drop('medv', axis=1)
y = boston_df.medv

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)


# In[26]:

# Pruning not supported. Choosing max depth 3)
regr2 = DecisionTreeRegressor(max_depth=3)
regr2.fit(X_train, y_train)
pred = regr2.predict(X_test)


# In[27]:

graph = print_tree(regr2, features=X.columns)
Image(graph.create_png())


# In[28]:

plt.scatter(pred, y_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')


# In[29]:

mean_squared_error(y_test, pred)


# ### 8.3.3 Bagging and Random Forests

# In[30]:

# There are 13 features in the dataset
X.shape


# In[31]:

# Bagging: using all features
regr1 = RandomForestRegressor(max_features=13, random_state=1)
regr1.fit(X_train, y_train)


# In[32]:

pred = regr1.predict(X_test)

plt.scatter(pred, y_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')


# In[33]:

mean_squared_error(y_test, pred)


# In[34]:

# Random forests: using 6 features
regr2 = RandomForestRegressor(max_features=6, random_state=1)
regr2.fit(X_train, y_train)


# In[35]:

pred = regr2.predict(X_test)
mean_squared_error(y_test, pred)


# In[36]:

Importance = pd.DataFrame({'Importance':regr2.feature_importances_*100}, index=X.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None


# ### 8.3.4 Boosting

# In[37]:

regr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, random_state=1)
regr.fit(X_train, y_train)


# In[38]:

feature_importance = regr.feature_importances_*100
rel_imp = pd.Series(feature_importance, index=X.columns).sort_values(inplace=False)
print(rel_imp)
rel_imp.T.plot(kind='barh', color='r', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None


# In[39]:

mean_squared_error(y_test, regr.predict(X_test))

