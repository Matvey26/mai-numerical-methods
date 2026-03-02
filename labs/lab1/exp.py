# from nm.linalg import (
#     thomas_algorithm,
# )
# from nm import Matrix

# A = [
#     [ 0, 3, 1],
#     [-1, 4, 2],
#     [ 2, 3, 0]
# ]
# b = [1, -3, 5]

# print(thomas_algorithm(A, b))

import numpy as np
from nm.linalg import lu_decomposition


A = np.array([
    [ 1,  2,  -1,  9],
    [ 1,  2,   1,  3],
    [ 5,  1,   8,  7],
    [-8,  6,   5,  1]
])

p, L, U = lu_decomposition(A)

print(p)
print(L)
print(U)
print('---')
print(L.dot(U))
print(A[p])
