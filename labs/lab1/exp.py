from nm.linalg import (
    thomas_algorithm,
)
from nm import Matrix

A = [
    [ 0, 3, 1],
    [-1, 4, 2],
    [ 2, 3, 0]
]
b = [1, -3, 5]

print(thomas_algorithm(A, b))