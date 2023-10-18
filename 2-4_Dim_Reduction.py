"""
2-4_Dim_Reduction.py

One of the most common applications of PCA is the visualisation of high-dimensional data sets.
As we have seen, it is difficult to create scatter plots of data that have more than two features.
For the iris dataset, we were able to create a matrix plot showing all possible combinations of two features.
On real data with dozens or hundreds of features, it is difficult to use such a plot.
With PCA we can find the first two principal components and visualise the data in this new two-dimensional space
with a single scatter plot.

Before applying PCA, we scale our data so that each feature has a unit variance using the StandardScaler.
Then we apply the PCA function with n_components=2 on the data set.
"""

import pandas as pd
import sklearn.preprocessing as pre
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Import the census data set again:
data = pd.read_csv('census.data', header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt',
                            'education', 'education-num', 'marital- status', 'occupation',
                            'relationship', 'race', 'gender', 'capital-gain',
                            'capital-loss', 'hours-per-week', 'native-country', 'income'])

# PCA is designed for numerical variables.
# It would not work well when we would binarize the categorical features.
# For the sake of this example, we are kicking out the categorical variables.
data.info()
    # We see that have 6 numerical variables in this data set, and all of them are predictors.
    # Our target variable is income, and it's categorical. We leave it in anyways (for the plot later on).
my_data = data[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'income']]

# Specify  predictors and target
X = my_data.loc[:,'age':'hours-per-week']  # predictors
y = my_data.loc[:,'income']  # target

# normalizing the data
scaler = pre.StandardScaler() # initialize
scaler.fit(X) # fit
X_scaled = scaler.transform(X) # transform

 # Apply PCA, keeping the first two principal components of the data
pca = PCA(n_components=2) # initialize
pca.fit(X_scaled) # fit
X_pca = pca.transform(X_scaled) # transform

# plot first vs. second principal component, colored by target class
plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pd.get_dummies(y)[' <=50K'].tolist())
    # Remark: To color the points by target class, we need to transform the categorical target values
    # to numbes. We do that by using get_dummies(). Since income is binary, it is sufficient to use
    # one of the resulting binary variables. I use ' <=50K'. Then convert it to a list for the parameter c to use.
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()

