"""
2-1_Binarization.py
How to make categorical variables numerical
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Import the census data set:
data = pd.read_csv('C:\\Users\\pc\\PycharmProjects\\MLprojects\\census.csv',
                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                          'marital-status', 'occupation', 'relationship', 'race', 'gender',
                          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])

# We select a subset of features
my_data = data[['age', 'workclass', 'gender', 'income']]

# Display the first few rows of the data
print(my_data.head())

# Display information about the DataFrame
print(my_data.info())

# Display summary statistics of the variables in the DataFrame
print(my_data.describe(include='all'))

# Transform categorical variables into binary variables
my_data_numerical = pd.get_dummies(my_data)
print(my_data_numerical.head())
print(my_data_numerical.info())
print(my_data_numerical.describe(include='all'))

# Convert the 'age' column to a string to treat it as categorical
my_data = my_data.copy()
my_data['age'] = my_data['age'].astype(str)
print(my_data.info())

# Apply get_dummies() to binarize all variables
my_data_numerical_gd = pd.get_dummies(my_data)
print(my_data_numerical_gd.info())
print(my_data_numerical_gd.head())

# Restore the original state of my_data
my_data = data[['age', 'workclass', 'gender', 'income']]
print(my_data.info())

# Use OneHotEncoder to binarize all features
onehot_data = OneHotEncoder(sparse=False)
my_data_numerical_ohe = onehot_data.fit_transform(my_data)
print(my_data_numerical_ohe)

# Convert the numpy array from OneHotEncoder to a DataFrame
df_onehot = pd.DataFrame(my_data_numerical_ohe, columns=onehot_data.get_feature_names_out(my_data.columns))
print(df_onehot.head())
