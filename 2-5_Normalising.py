"""
2-5_Normalizing.py

The simplest way to normalise numerical features is to use the MinMaxScaler or the StandardScaler
from the sklearn.preprocessing module.

The MinMaxScaler uses the minimum and maximum.
The StandardScaler is based on the mean and standard deviation of the data.

Note that the normalisation again proceeds in 2 steps, fit and transform.
"""

import sklearn.datasets as ds
import sklearn.preprocessing as pre
import pandas

# load the integrated dataset iris
iris = ds.load_iris()
    # Remember that iris is a bunch, not a data frame.
    # The data is stored in the feature 'data' as an ndarray:
X = iris["data"]

# We normalize with the min-max method:
min_max_scaler = pre.MinMaxScaler() # initialize
min_max_scaler.fit(X) # fit
X_norm_min_max = min_max_scaler.transform(X) # transform

# We normalize with the mu-sigma method:
mu_sigma_scaler = pre.StandardScaler() # initialize
mu_sigma_scaler.fit(X) # fit
X_norm_mu_sigma = mu_sigma_scaler.transform(X) # transform