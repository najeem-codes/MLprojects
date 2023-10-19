import numpy as np

# one-dimensional array
a = np.array([1, 2, 3, 4, 5, 6])
print(a[3]) # 4
print(a[2:5]) # [3 4 5]
# two-dimensional array
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a[0]) # [1 2 3 4]
print(a[2][3]) # 12


'''In addi:on, NumPy arrays offer a few aLributes (number of dimensions, elements and size of 
the data set):'''

# attributes on an array a
print(a.ndim) # number of dimensions = 2 print(a.size) # number of elements = 12
print(a.shape) # Size of the data set (rows, columns) = (3, 4)

'The biggest difference to lists (in terms of syntax) is the access to whole rows or columns:'

# whole row or parts of rows from an array a
row1 = a[1, :] # equivalent to: row1 = a[1]
print(row1) # [5 6 7 8]
part_of_row1 = a[1, 1:3]
print(part_of_row1) # [6 7]
# complete columns from the array a (not possible on lists!)
col0 = a[:, 0]
print(col0) # [1 5 9]