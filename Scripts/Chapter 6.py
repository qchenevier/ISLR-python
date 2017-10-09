
# coding: utf-8

# - [Lab 2: Ridge Regression](#6.6.1-Ridge-Regression)
# - [Lab 2: The Lasso](#6.6.2-The-Lasso)
# - [Lab 3: Principal Components Regression](#6.7.1-Principal-Components-Regression)
# - [Lab 3: Partial Least Squares](#6.7.2-Partial-Least-Squares)

# # Chapter 6 - Linear Model Selection and Regularization

# In[1]:

# %load ../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glmnet as gln

from sklearn.preprocessing import scale 
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

pd.set_option('display.notebook_repr_html', False)

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-white')


# # Lab 2

# In[2]:

# In R, I exported the dataset from package 'ISLR' to a csv file.
df = pd.read_csv('Data/Hitters.csv', index_col=0).dropna()
df.index.name = 'Player'
df.info()


# In[3]:

df.head()


# In[4]:

dummies = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
dummies.info()
print(dummies.head())


# In[5]:

y = df.Salary

# Drop the column with the independent variable (Salary), and columns for which we created dummy variables
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
# Define the feature set X.
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X.info()


# In[6]:

X.head(5)


# #### I executed the R code and downloaded the exact same training/test sets used in the book.

# In[7]:

X_train = pd.read_csv('data/Hitters_X_train.csv', index_col=0)
y_train = pd.read_csv('data/Hitters_y_train.csv', index_col=0)
X_test = pd.read_csv('data/Hitters_X_test.csv', index_col=0)
y_test = pd.read_csv('data/Hitters_y_test.csv', index_col=0)


# ### 6.6.1 Ridge Regression

# ### Scikit-learn

# The __glmnet__ algorithms in R optimize the objective function using cyclical coordinate descent, while scikit-learn Ridge regression uses linear least squares with L2 regularization. They are rather different implementations, but the general principles are the same.
# 
# For the __glmnet() function in R__ the penalty is defined as:
# ### $$ \lambda\bigg(\frac{1}{2}(1−\alpha)|\beta|^2_2 \ +\ \alpha|\beta|_1\bigg) $$
# (See R documentation and https://cran.r-project.org/web/packages/glmnet/vignettes/glmnet_beta.html)<BR>
# The function supports L1 and L2 regularization. For just Ridge regression we need to use $\alpha = 0 $. This reduces the above penalty to
# ### $$ \frac{1}{2}\lambda |\beta|^2_2 $$
# The __sklearn Ridge()__ function has the standard L2 penalty:
# ### $$ \lambda |\beta|^2_2 $$
# 

# In[8]:

alphas = 10**np.linspace(10,-2,100)*0.5

ridge = Ridge()
coefs = []

for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(scale(X), y)
    coefs.append(ridge.coef_)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization');


# The above plot shows that the Ridge coefficients get larger when we decrease alpha.

# #### Alpha = 4

# In[9]:

ridge2 = Ridge(alpha=4)
ridge2.fit(scale(X_train), y_train)
pred = ridge2.predict(scale(X_test))
mean_squared_error(y_test, pred)


# #### Alpha = $10^{10}$ 
# This big penalty shrinks the coefficients to a very large degree and makes the model more biased, resulting in a higher MSE.

# In[10]:

ridge2.set_params(alpha=10**10)
ridge2.fit(scale(X_train), y_train)
pred = ridge2.predict(scale(X_test))
mean_squared_error(y_test, pred)


# #### Compute the regularization path using RidgeCV

# In[11]:

ridgecv = RidgeCV(alphas=alphas, scoring='mean_squared_error')
ridgecv.fit(scale(X_train), y_train)


# In[12]:

ridgecv.alpha_


# In[13]:

ridge2.set_params(alpha=ridgecv.alpha_)
ridge2.fit(scale(X_train), y_train)
mean_squared_error(y_test, ridge2.predict(scale(X_test)))


# In[14]:

pd.Series(ridge2.coef_.flatten(), index=X.columns)


# ### python-glmnet (update 2016-08-29)
# This relatively new module is a wrapper for the fortran library used in the R package `glmnet`. It gives mostly the exact same results as described in the book. However, the `predict()` method does not give you the regression *coefficients* for lambda values not in the lambda_path. It only returns the predicted values.
# https://github.com/civisanalytics/python-glmnet

# In[15]:

grid = 10**np.linspace(10,-2,100)

ridge3 = gln.ElasticNet(alpha=0, lambda_path=grid)
ridge3.fit(X, y)


# #### Lambda 11498

# In[16]:

ridge3.lambda_path_[49]


# In[17]:

print('Intercept: {:.3f}'.format(ridge3.intercept_path_[49]))


# In[18]:

pd.Series(np.round(ridge3.coef_path_[:,49], decimals=3), index=X.columns)


# In[19]:

np.sqrt(np.sum(ridge3.coef_path_[:,49]**2))


# #### Lambda 705

# In[20]:

ridge3.lambda_path_[59]


# In[21]:

print('Intercept: {:.3f}'.format(ridge3.intercept_path_[59]))


# In[22]:

pd.Series(np.round(ridge3.coef_path_[:,59], decimals=3), index=X.columns)


# In[23]:

np.sqrt(np.sum(ridge3.coef_path_[:,59]**2))


# #### Fit model using just the training set.

# In[24]:

ridge4 = gln.ElasticNet(alpha=0, lambda_path=grid, scoring='mean_squared_error', tol=1e-12)
ridge4.fit(X_train, y_train.as_matrix().ravel())


# In[25]:

# prediction using lambda = 4
pred = ridge4.predict(X_test, lamb=4)
mean_squared_error(y_test.as_matrix().ravel(), pred)


# #### Lambda chosen by cross validation

# In[26]:

ridge5 = gln.ElasticNet(alpha=0, scoring='mean_squared_error')
ridge5.fit(X_train, y_train.as_matrix().ravel())


# In[27]:

# Lambda with best CV performance
ridge5.lambda_max_


# In[28]:

# Lambda larger than lambda_max_, but with a CV score that is within 1 standard deviation away from lambda_max_ 
ridge5.lambda_best_


# In[29]:

plt.figure(figsize=(15,6))
plt.errorbar(np.log(ridge5.lambda_path_), -ridge5.cv_mean_score_, color='r', linestyle='None', marker='o',
             markersize=5, yerr=ridge5.cv_standard_error_, ecolor='lightgrey', capthick=2)

for ref, txt in zip([ridge5.lambda_best_, ridge5.lambda_max_], ['Lambda best', 'Lambda max']):
    plt.axvline(x=np.log(ref), linestyle='dashed', color='lightgrey')
    plt.text(np.log(ref), .95*plt.gca().get_ylim()[1], txt, ha='center')

plt.xlabel('log(Lambda)')
plt.ylabel('Mean-Squared Error');


# In[30]:

# MSE for lambda with best CV performance
pred = ridge5.predict(X_test, lamb=ridge5.lambda_max_)
mean_squared_error(y_test, pred)


# #### Fit model to full data set

# In[31]:

ridge6= gln.ElasticNet(alpha=0, scoring='mean_squared_error', n_folds=10)
ridge6.fit(X, y)


# In[32]:

# These are not really close to the ones in the book.
pd.Series(ridge6.coef_path_[:,ridge6.lambda_max_inx_], index=X.columns)


# ### 6.6.2 The Lasso

# ### Scikit-learn

# 
# 
# For both __glmnet__ in R and sklearn __Lasso()__ function the standard L1 penalty is:
# ### $$ \lambda |\beta|_1 $$

# In[33]:

lasso = Lasso(max_iter=10000)
coefs = []

for a in alphas*2:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Lasso coefficients as a function of the regularization');


# In[34]:

lassocv = LassoCV(alphas=None, cv=10, max_iter=10000)
lassocv.fit(scale(X_train), y_train.as_matrix().ravel())


# In[35]:

lassocv.alpha_


# In[36]:

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(scale(X_train), y_train)
mean_squared_error(y_test, lasso.predict(scale(X_test)))


# In[37]:

# Some of the coefficients are now reduced to exactly zero.
pd.Series(lasso.coef_, index=X.columns)


# ### python-glmnet

# In[38]:

lasso2 = gln.ElasticNet(alpha=1, lambda_path=grid, scoring='mean_squared_error', n_folds=10)
lasso2.fit(X_train, y_train.as_matrix().ravel())


# In[39]:

l1_norm = np.sum(np.abs(lasso2.coef_path_), axis=0)

plt.figure(figsize=(10,6))
plt.plot(l1_norm, lasso2.coef_path_.T)
plt.xlabel('L1 norm')
plt.ylabel('Coefficients');


# #### Let glmnet() create a grid to use in CV

# In[40]:

lasso3 = gln.ElasticNet(alpha=1, scoring='mean_squared_error', n_folds=10)
lasso3.fit(X_train, y_train.as_matrix().ravel())


# In[41]:

plt.figure(figsize=(15,6))
plt.errorbar(np.log(lasso3.lambda_path_), -lasso3.cv_mean_score_, color='r', linestyle='None', marker='o',
             markersize=5, yerr=lasso3.cv_standard_error_, ecolor='lightgrey', capthick=2)

for ref, txt in zip([lasso3.lambda_best_, lasso3.lambda_max_], ['Lambda best', 'Lambda max']):
    plt.axvline(x=np.log(ref), linestyle='dashed', color='lightgrey')
    plt.text(np.log(ref), .95*plt.gca().get_ylim()[1], txt, ha='center')

plt.xlabel('log(Lambda)')
plt.ylabel('Mean-Squared Error');


# In[42]:

pred = lasso3.predict(X_test, lamb=lasso3.lambda_max_)
mean_squared_error(y_test, pred)


# #### Fit model on full dataset

# In[43]:

lasso4 = gln.ElasticNet(alpha=1, lambda_path=grid, scoring='mean_squared_error', n_folds=10)
lasso4.fit(X, y)


# In[44]:

# These are not really close to the ones in the book.
pd.Series(lasso4.coef_path_[:,lasso4.lambda_max_inx_], index=X.columns)


# # Lab 3

# ### 6.7.1 Principal Components Regression

# Scikit-klearn does not have an implementation of PCA and regression combined like the 'pls' package in R.
# https://cran.r-project.org/web/packages/pls/vignettes/pls-manual.pdf

# In[45]:

pca = PCA()
X_reduced = pca.fit_transform(scale(X))

print(pca.components_.shape)
pd.DataFrame(pca.components_.T).loc[:4,:5]


# The above loadings are the same as in R.

# In[46]:

print(X_reduced.shape)
pd.DataFrame(X_reduced).loc[:4,:5]


# The above principal components are the same as in R.

# In[47]:

# Variance explained by the principal components
np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


# In[54]:

# 10-fold CV, with shuffle
n = len(X_reduced)
kf_10 = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=1)

regr = LinearRegression()
mse = []

# Calculate MSE with only the intercept (no principal components in regression)
score = -1*cross_validation.cross_val_score(regr, np.ones((n,1)), y.ravel(), cv=kf_10, scoring='mean_squared_error').mean()    
mse.append(score)

# Calculate MSE using CV for the 19 principle components, adding one component at the time.
for i in np.arange(1, 20):
    score = -1*cross_validation.cross_val_score(regr, X_reduced[:,:i], y.ravel(), cv=kf_10, scoring='mean_squared_error').mean()
    mse.append(score)
    
plt.plot(mse, '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('Salary')
plt.xlim(xmin=-1);


# The above plot indicates that the lowest training MSE is reached when doing regression on 18 components.

# In[55]:

regr_test = LinearRegression()
regr_test.fit(X_reduced, y)
regr_test.coef_


# #### Fitting PCA with training data

# In[56]:

pca2 = PCA()
X_reduced_train = pca2.fit_transform(scale(X_train))
n = len(X_reduced_train)

# 10-fold CV, with shuffle
kf_10 = cross_validation.KFold(n, n_folds=10, shuffle=False, random_state=1)

mse = []

# Calculate MSE with only the intercept (no principal components in regression)
score = -1*cross_validation.cross_val_score(regr, np.ones((n,1)), y_train, cv=kf_10, scoring='mean_squared_error').mean()    
mse.append(score)

# Calculate MSE using CV for the 19 principle components, adding one component at the time.
for i in np.arange(1, 20):
    score = -1*cross_validation.cross_val_score(regr, X_reduced_train[:,:i], y_train, cv=kf_10, scoring='mean_squared_error').mean()
    mse.append(score)

plt.plot(np.array(mse), '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('Salary')
plt.xlim(xmin=-1);


# The above plot indicates that the lowest training MSE is reached when doing regression on 6 components.

# #### Transform test data with PCA loadings and fit regression on 6 principal components

# In[57]:

X_reduced_test = pca2.transform(scale(X_test))[:,:7]

# Train regression model on training data 
regr = LinearRegression()
regr.fit(X_reduced_train[:,:7], y_train)

# Prediction with test data
pred = regr.predict(X_reduced_test)
mean_squared_error(y_test, pred)


# ### 6.7.2 Partial Least Squares

# Scikit-learn PLSRegression gives same results as the pls package in R when using 'method='oscorespls'. In the LAB excercise, the standard method is used which is 'kernelpls'. 
# 
# When doing a slightly different fitting in R, the result is close to the one obtained using scikit-learn.
# 
#     pls.fit=plsr(Salary~., data=Hitters, subset=train, scale=TRUE, validation="CV", method='oscorespls')
#     validationplot(pls.fit,val.type="MSEP", intercept = FALSE)
#    
# See documentation:
# http://scikit-learn.org/dev/modules/generated/sklearn.cross_decomposition.PLSRegression.html#sklearn.cross_decomposition.PLSRegression

# In[58]:

n = len(X_train)

# 10-fold CV, with shuffle
kf_10 = cross_validation.KFold(n, n_folds=10, shuffle=False, random_state=1)

mse = []

for i in np.arange(1, 20):
    pls = PLSRegression(n_components=i)
    score = cross_validation.cross_val_score(pls, scale(X_train), y_train, cv=kf_10, scoring='mean_squared_error').mean()
    mse.append(-score)

plt.plot(np.arange(1, 20), np.array(mse), '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('Salary')
plt.xlim(xmin=-1);


# In[59]:

pls = PLSRegression(n_components=2)
pls.fit(scale(X_train), y_train)

mean_squared_error(y_test, pls.predict(scale(X_test)))

