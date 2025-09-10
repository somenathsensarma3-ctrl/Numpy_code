# Numpy_code by Somenath sen Sarma
import numpy as np

# ============================================================
# 1. Create a 5×5 matrix with values 1,2,3,…,25
# ============================================================
matrix = np.arange(1, 26).reshape(5, 5)
print("Q1: 5x5 Matrix\n", matrix, "\n")

# ============================================================
# 2. Generate a 4×4 identity matrix
# ============================================================
I = np.eye(4)
print("Q2: 4x4 Identity Matrix\n", I, "\n")

# ============================================================
# 3. Create a 1D array of numbers from 100 to 200 with step size 10
# ============================================================
arr = np.arange(100, 201, 10)
print("Q3: Array from 100 to 200 step 10\n", arr, "\n")

# ============================================================
# 4. Random 3×3 matrix and find its determinant
# ============================================================
A = np.random.rand(3, 3)
det = np.linalg.det(A)
print("Q4: Random 3x3 Matrix\n", A)
print("Determinant:", det, "\n")

# ============================================================
# 5. 10 random integers between 1 and 100
# ============================================================
rand_ints = np.random.randint(1, 101, 10)
print("Q5: 10 Random Integers\n", rand_ints, "\n")

# ============================================================
# 6. Reshape 1D array of size 12 into 3×4
# ============================================================
arr = np.arange(1, 13)
reshaped = arr.reshape(3, 4)
print("Q6: Reshaped Array\n", reshaped, "\n")

# ============================================================
# 7. Matrix multiplication of two 3×3 matrices
# ============================================================
A = np.random.randint(1, 10, (3, 3))
B = np.random.randint(1, 10, (3, 3))
product = np.dot(A, B)
print("Q7: A\n", A)
print("B\n", B)
print("A*B\n", product, "\n")

# ============================================================
# 8. Eigenvalues and eigenvectors of 2×2 matrix
# ============================================================
M = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(M)
print("Q8: Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors, "\n")

# ============================================================
# 9. Extract diagonal elements of 5×5 random matrix
# ============================================================
M = np.random.rand(5, 5)
diag = np.diag(M)
print("Q9: Diagonal elements\n", diag, "\n")

# ============================================================
# 10. Normalize 1D array (0–1)
# ============================================================
arr = np.array([10, 20, 30, 40, 50])
norm = (arr - arr.min()) / (arr.max() - arr.min())
print("Q10: Normalized array\n", norm, "\n")

# ============================================================
# 11. Sort array by row and column
# ============================================================
A = np.random.randint(1, 50, (4, 4))
print("Q11: Original\n", A)
print("Row-wise sort\n", np.sort(A, axis=1))
print("Column-wise sort\n", np.sort(A, axis=0), "\n")

# ============================================================
# 12. Indices of max and min values
# ============================================================
arr = np.random.randint(1, 100, 10)
print("Q12: Array\n", arr)
print("Max index:", arr.argmax(), "Min index:", arr.argmin(), "\n")

# ============================================================
# 13. Flatten 2D array
# ============================================================
A = np.array([[1, 2], [3, 4]])
print("Q13: ravel:", A.ravel(), " flatten:", A.flatten(), "\n")

# ============================================================
# 14. Inverse of 3×3 matrix
# ============================================================
A = np.array([[1, 2, 1], [0, 1, 0], [2, 3, 4]])
inv = np.linalg.inv(A)
print("Q14: Inverse\n", inv, "\n")

# ============================================================
# 15. Random permutation of 1–10
# ============================================================
perm = np.random.permutation(np.arange(1, 11))
print("Q15: Permutation\n", perm, "\n")

# ============================================================
# 16. Replace even numbers with -1
# ============================================================
arr = np.arange(21)
arr[arr % 2 == 0] = -1
print("Q16: Replace evens with -1\n", arr, "\n")

# ============================================================
# 17. Dot product of two arrays
# ============================================================
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("Q17: Dot product:", np.dot(a, b), "\n")

# ============================================================
# 18. Trace of 5×5 random matrix
# ============================================================
A = np.random.rand(5, 5)
print("Q18: Trace:", np.trace(A), "\n")

# ============================================================
# 19. Split 1D array into 3 equal parts
# ============================================================
arr = np.arange(9)
parts = np.split(arr, 3)
print("Q19: Split array:", parts, "\n")

# ============================================================
# 20. Mean across axis=0 of 3D array
# ============================================================
arr = np.random.rand(3, 3, 3)
print("Q20: Mean across axis=0\n", arr.mean(axis=0), "\n")

# ============================================================
# 21. Cumulative sum
# ============================================================
arr = np.array([1, 2, 3, 4])
print("Q21: Cumulative sum:", np.cumsum(arr), "\n")

# ============================================================
# 22. Upper triangular matrix
# ============================================================
A = np.random.randint(1, 10, (4, 4))
upper = np.triu(A)
print("Q22: Upper triangular\n", upper, "\n")

# ============================================================
# 23. Checkerboard pattern 6×6
# ============================================================
checker = np.indices((6, 6)).sum(axis=0) % 2
print("Q23: Checkerboard pattern\n", checker, "\n")

# ============================================================
# 24. Element-wise sqrt of 3×3 matrix
# ============================================================
A = np.random.rand(3, 3) * 10
print("Q24: Square root\n", np.sqrt(A), "\n")

# ============================================================
# 25. Reverse 1D array of 20 elements
# ============================================================
arr = np.arange(20)
rev = np.flip(arr)
print("Q25: Reversed array\n", rev, "\n")

# ============================================================
# 26. Merge arrays vertically and horizontally
# ============================================================
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print("Q26: Vertical\n", np.vstack((a, b)))
print("Horizontal\n", np.hstack((a, b)), "\n")

# ============================================================
# 27. Row-wise and column-wise sum
# ============================================================
A = np.array([[1, 2, 3], [4, 5, 6]])
print("Q27: Row sum", A.sum(axis=1))
print("Column sum", A.sum(axis=0), "\n")

# ============================================================
# 28. Replace NaN with column mean
# ============================================================
arr = np.array([[1, np.nan, 3], [4, 5, np.nan]])
col_mean = np.nanmean(arr, axis=0)
inds = np.where(np.isnan(arr))
arr[inds] = np.take(col_mean, inds[1])
print("Q28: Replace NaN\n", arr, "\n")

# ============================================================
# 29. Cosine similarity
# ============================================================
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print("Q29: Cosine similarity:", cos_sim, "\n")

# ============================================================
# 30. Rotate 4×4 array by 90°
# ============================================================
A = np.arange(16).reshape(4, 4)
rot = np.rot90(A)
print("Q30: Rotated\n", rot, "\n")

# ============================================================
# 31. Structured array
# ============================================================
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('marks', 'f4')])
students = np.array([('Alice', 21, 88.5), ('Bob', 22, 92.0)], dtype=dt)
print("Q31: Structured array\n", students, "\n")

# ============================================================
# 32. Rank of random 3×3 matrix
# ============================================================
A = np.random.rand(3, 3)
print("Q32: Rank:", np.linalg.matrix_rank(A), "\n")

# ============================================================
# 33. Normalize each row to unit length
# ============================================================
A = np.random.rand(5, 5)
norms = np.linalg.norm(A, axis=1, keepdims=True)
normalized = A / norms
print("Q33: Normalized rows\n", normalized, "\n")

# ============================================================
# 34. Check element-wise equality
# ============================================================
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
print("Q34: Arrays equal?", np.array_equal(a, b), "\n")

# ============================================================
# 35. Histogram of 1000 random numbers
# ============================================================
data = np.random.randn(1000)
hist, bins = np.histogram(data, bins=10)
print("Q35: Histogram\n", hist)
print("Bins\n", bins, "\n")

# ============================================================
# 36. Broadcasting with 2D and 1D array
# ============================================================
A = np.ones((3, 3))
b = np.array([1, 2, 3])
print("Q36: Broadcasting\n", A + b, "\n")

# ============================================================
# 37. Unique values and counts
# ============================================================
arr = np.array([1, 2, 2, 3, 4, 4, 4])
uniq, counts = np.unique(arr, return_counts=True)
print("Q37: Unique\n", uniq, "Counts\n", counts, "\n")

# ============================================================
# 38. Pearson correlation coefficient
# ============================================================
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
corr = np.corrcoef(x, y)[0, 1]
print("Q38: Pearson correlation:", corr, "\n")

# ============================================================
# 39. Numerical gradient of 1D array
# ============================================================
arr = np.array([1, 2, 4, 7, 11])
grad = np.gradient(arr)
print("Q39: Gradient\n", grad, "\n")

# ============================================================
# 40. Singular Value Decomposition (SVD)
# ============================================================
A = np.random.rand(3, 3)
U, S, Vt = np.linalg.svd(A)
print("Q40: U\n", U)
print("S\n", S)
print("Vt\n", Vt, "\n")
