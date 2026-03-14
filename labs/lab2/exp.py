import numpy as np
from scipy.sparse import csr_matrix
from nm.linalg import simple_iterations, seidel_method


A = csr_matrix([
    [21, 1, -8, 4],
    [-9, -23, -2, 4],
    [7, -1, -17, 6],
    [8, 8, -4, -26]
])

b = np.array([-119, 79, -24, -52])

x = simple_iterations(A, b, precision_epsilon=1e-3)
print(A.dot(x) - b)

print()

x = seidel_method(A, b, precision_epsilon=1e-3)
print(A.dot(x) - b)