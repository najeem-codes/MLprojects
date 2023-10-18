"""
2-3_Feature_Selection.py

How to reduce the number of features.

Notice: The goal is to do it in a way so that the remaining features give
the best results for our machine learning algorithm.

There are 2 fundamental approaches: filter methods and wrapper methods.
We give one example for each of them.
"""

import pandas as pd
import numpy as np
import sklearn.feature_selection as fs
import matplotlib.pyplot as plt
import sklearn.datasets as ds

# We generate synthetic data with 60 features
X, y = ds.make_classification(n_samples=100, n_features=60, n_redundant=0, n_classes=2)

######## Applying a filter method for feature selection
# In a filter method, features are evaluated independently of the machine learning algorithm that we intend to apply.
# For each feature, we define a quality score.
# The better a feature scores, the higher it is ranked. We then choose the highest ranked k features.
# In this example, we use as a quality score a metric that measures the strength of relationship between a feature
# and the target variable. The metric we use is called ANOVA F-value.

# We use teh function SelectKBest() to select k=15 individual best features out of the givn 60 features
# The parameter score_func defines the quality score. f_classif is a function that returns the ANOVA F-value
# for a feature in X and the target y.
select_kb = fs.SelectKBest(score_func=fs.f_classif, k=15) # initialize
select_kb.fit(X, y) # fit
X_selected_kb = select_kb.transform(X) # transform (i.e., apply)

# visualize selection (black is selected, white is not selected)
mask = select_kb.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Feature Index")
plt.show()

######## Applying a wrapper method for feature selection
# A wrapper method uses a specific machine learning algorithm to evaluate the quality of a feature set.
# The idea is train the algorithm on the feature set and to then evaluate how good the algorithm performs.
# The algorithms performance on the feature set is used as a quality measure for this feature set.
# In this example, we use the a simple classification algorithm, the KNeighborsClassifier.
# We evaluate it's perfomrance on a specific feature set as using the recognition rate.
# As a search strategy, we use Sequential Foreward Search (sfs).

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
select_sfs = fs.SequentialFeatureSelector(estimator=knn, n_features_to_select=15) # initialize
select_sfs.fit(X, y) # fit
X_selected_sfs = select_sfs.transform(X) # transform (i.e., apply)
