import numpy as np

# 1. Create Arrays
a1 = np.array([1, 2, 3])
a2 = np.array([4, 5, 6])

m1 = np.array([[10, 20, 30], [1, 2, 3]])
m2 = np.array([[4, 5, 6], [7, 8, 9]])

# 2. Arithmetic Operations

print("Addition:", a1 + a2)
print("Subtraction:", a1 - a2)
print("Multiplication:", a1 * a2)
print("Power:", a1 ** 2)

print("Matrix Add:\n", np.add(m1, m2))
print("Matrix Multiply:\n", np.multiply(m1, m2))

# 3. Statistical Functions

print("Mean:", np.mean(a1))
print("Median (row):", np.median(m1, axis=1))
print("Median (column):", np.median(m1, axis=0))
print("Standard Deviation:", np.std(m1))

# 4. Min & Max
arr = np.array([2, 6, 9, 15, 18, 1, 47, 65, 27])
print("Minimum value:", np.min(arr))
print("Maximum value:", np.max(arr))

# 5. Reshape & Flatten
x = np.array([[1,2,3,1],
              [4,5,6,1],
              [7,8,9,1]])

print("Original array:\n", x)

new_arr = x.reshape(2,2,3)
print("Reshaped array:\n", new_arr)

flat = x.flatten()
print("Flatten array:", flat)