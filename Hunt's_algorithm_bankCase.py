"""
3-1_Decision_Tree.py
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt

# Load the iris data set
iris = datasets.load_iris()
X = iris.data
y = iris.target

# First, we split the data into a training and a test set.
# For this we use the function train_test_split.
# By default, 25% of the data is moved to the test set
# and 75% of the data to the training set (this can be adjusted with the parameter test_size).
# In order to preserve the class distribution when splitting between the two sets,
# one can set the parameter stratify=y (y references the class).
# The train_test_split takes random samples from the data set. If you want to repeat it later in exactly the same way,
# you can make the random choice reproducible with random_state=23 (23 is an arbitrary value).

# split the data to training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.4, random_state=2)

# The following code example learns a decision tree on the training set (with the function fit).
# The Gini index is used as a criterion to partition the training set (criterion='gini').
# Then we apply the learned tree on the training set and the test set (using the score function
# to measure the recognition rate).

# learn unpruned tree
clf = DecisionTreeClassifier(criterion='gini', random_state=0)
model = clf.fit(X_train, y_train)
print("Accuracy on training set:",  clf.score(X_train, y_train))
print("Accuracy on test set:", clf.score(X_test, y_test))

# As expected, the accuracy on the training set is 100% - since the leaves are pure,
# the tree was derived deep enough so that it could perfectly imprint all classes on the training data.
# The accuracy on the test set in this example is about 91.6%.

# The following code visualizes the tree.
# In your SciView window, go to the Plots tab to see it.
# You may need to expand the window.

# visualize the tree (open in onmigraffle or similar)
fig = plt.figure(figsize=(25,20))
plot_tree(clf,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          rounded=True, # Rounded node edges
          filled=True, # Adds color according to class
          proportion=True); # Displays the proportions of class samples instead of the whole number of samples
plt.show()

# Next, we prune the tree - there are various possibilities for this.
# For example, a set of data objects is only further divided if this division
# causes a reduction in impurity greater than or equal to the parameter min_impurity_decrease.
# Another possibility is to end the construction of the building after reaching a certain depth.
# Here, for example, we set max_depth=3, which means that a maximum of 3 consecutive questions can be asked:

# pruning: max 3 levels
clf_p = DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=3)
model_p = clf_p.fit(X_train, y_train)
print("Accuracy on training set:",  clf_p.score(X_train, y_train))
print("Accuracy on test set:", clf_p.score(X_test, y_test))

# Obviously, stopping the splitting early has led to a lower accuracy on the training set (96.6%).
# The accuracy on the test set indeed improved from 91.6% to 95%.

# visualize the tree
fig = plt.figure(figsize=(25,20))
plot_tree(clf_p,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          rounded=True, # Rounded node edges
          filled=True, # Adds color according to class
          proportion=True); # Displays the proportions of class samples instead of the whole number of samples
plt.show()

# Finding the best values of the parameters of a model (those that give the best accuracy) is a difficult task,
# but necessary for almost all models. It is important to understand what the parameters mean before trying to
# adjust them.
# A commonly used method is grid search, which basically means systematically trying different combinations
# of the parameters of interest. We can program a simple grid search with a for loop.
# With nested for-loops, for example, you can learn and evaluate a model for each combination of parameters.
# The following code looks for a good parameter combination for min_impurity_decrease and max_depth:


# validation of parameters via grid search
best_score = 0
best_parameters = {}
for x in [0.0001, 0.001, 0.01, 0.1]:
    for y in range(3, 8, 2):
        clf_opt = DecisionTreeClassifier(random_state=0,
                                       min_impurity_decrease=x,
                                       max_depth=y)
        clf_opt.fit(X_train, y_train)
        score = clf_opt.score(X_test, y_test)
        # if we got a better score, store the score and parameters
        if score >= best_score:
            best_score = score
            best_parameters = {"min_impurity_decrease": x, "max_depth": y}

print("Best score:", best_score)
print("Best parameters:", best_parameters)

# In this example, the best parameter combination is 'min_impurity_decrease': 0.01, 'max_depth': 7,
# with an accuracy of 98.3% on the test set.
# This is not better than our first attempt without pruning.

# Let's look at the accuracy on the training set:
print("Accuracy on training set:",  clf_opt.score(X_train, y_train))

# Compared to the pruned tree, the accuracies on the training and test set did not change.
# Let's look at the visualization:

# visualize the tree (open in onmigraffle or similar)
fig = plt.figure(figsize=(25,20))
plot_tree(clf_opt,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          rounded=True, # Rounded node edges
          filled=True, # Adds color according to class
          proportion=True); # Displays the proportions of class samples instead of the whole number of samples
plt.show()

# Even though the accuracy values did not change, we can see that the tree is smaller than our first pruning attempt.
# Smaller trees are preferable, because they are is easier to interpret.
# Yet, in this case, our sample data set is quite small (150 data objects), and the output is quite senitive to
# a change in the training / test data configuration. E.g., try to change the seed for the randomizer in the
# function train_test_split to a different number, e.g., random_state=2, and you will see that the accuracy values
# are quite different.
# Notice that the parameter random_state is only a seed for the pseudo-randomizer and does not change our learning
# algorithm. Only the random division in training and test set ois changed.