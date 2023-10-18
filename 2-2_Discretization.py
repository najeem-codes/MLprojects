"""
2-1_Discretization.py
How to make numerical variables categorical

Discretisation of numerical features is most simply done via the KBinsDiscretizer
from the sklearn.preprocessing module.
Here you specify how many categories your numerical data should be divided into (n_bins)
and whether you want to discretise with Equal-Width (strategy='uniform') or
Equal-Frequency (strategy='quantile') binning.

The discretisation takes place in two steps or with two function calls:
• fit
• transform

In the first step, the categorisation is "learned" and only in the second step is it actually carried out.
This has the advantage that we can also apply the learned transformation to new data.
This principle applies to almost all algorithms from the scikit-learn project.

In the following programme, a synthetically generated data set is discretised once with
Equal Width and once with Equal Frequency binning.
"""

import sklearn.datasets as ds
import sklearn.preprocessing as pre

# Generate a synthetic data set with 3 features and 10 data objects
X, y = ds.make_classification(n_samples=10, n_features=3, n_redundant=0, n_classes=2)

# Print the input variables X
print(X)

# We perform Equal Width Binning (i.e., all intervals have the same width.)
# To do this we use KBinsDiscretizer() and set the parameter strategy='uniform':
est_ew = pre.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')  # initialize
est_ew.fit(X)  # fit
X_ew_binning = est_ew.transform(X)  # transform

# Look at the transformed variables:
print(X_ew_binning)

# We can inspect the interval bounderies with bin_edges_:
est_ew.bin_edges_
    # bin_edges_ returns all interval bounderies per variable.
    # Since we decided to have 3 bins (intervals), we get 4 bin edges (interval boundaries).
    # Notice that each of the variables is dicretized seperately.
    # Therefore we get one array of split points (interval boundaries) per variable.

# We perform Equal Frequency Binning with strategy='quantile'.
# All intervals contain the same number of data objects.
est_ef = pre.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')  # initialize
est_ef.fit(X)  # fit
X_ef_binning = est_ef.transform(X)  # transform

# Look at the transformed variables:
print(X_ef_binning)

# Inspect the bin edges like this:
temp = est_ef.bin_edges_[1]
    # Compare this to est_ew.bin_edges_.
    # We see that some of them are different.

