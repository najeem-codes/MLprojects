

import sklearn as ds
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


# 100 data objects from two classes described by two features
# X = features, y = target (class)
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_classes=2)
# output the features and the target
print(X)
print(y)