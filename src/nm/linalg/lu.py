"""
https://stepanzh.github.io/computational_thermodynamics/syslinear/plu.html
"""
import numpy as np
from ..utils import count_permutation


def lu_decomposition(A: np.ndarray, epsilon: float = 1e-8) -> tuple[list[int], np.ndarray, np.ndarray]:

    if A.ndim != 2 and A.shape[0] != A.shape[1]:
        raise ValueError('Матрица A должна быть квадратной')

    A = A.astype(float)
    n = A.shape[0]

    U = np.zeros(A.shape)
    L = np.zeros(A.shape)
    p = list(range(n))

    A_k = A.copy()

    for k in range(n):
        pivot_idx = max(range(k, n), key=lambda i: abs(A_k[i, k]))

        if pivot_idx != k:
            A_k[[pivot_idx, k]] = A_k[[k, pivot_idx]]
            L[[pivot_idx, k]] = L[[k, pivot_idx]]
            p[pivot_idx], p[k] = p[k], p[pivot_idx]

        if abs(A_k[k, k]) < epsilon:
            raise ValueError('Матрица A плохо обусловлена (или вырождена)')

        U[k, k:] = A_k[k, k:]
        L[k:, k] = A_k[k:, k] / U[k, k]
        A_k[k:, k:] -= np.outer(L[k:, k], U[k, k:])

    return p, L, U


def forward_pass(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = b.shape[0]
    x = np.zeros_like(b)
    x[0] = b[0] / L[0, 0]
    for i in range(1, n):
        x[i] = (b[i] - L[i, :i].dot(x[:i]).item()) / L[i, i]
    
    return x


def backward_pass(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = b.shape[0]
    x = np.zeros_like(b)
    x[n - 1] = b[n - 1] / U[n - 1, n - 1]
    for i in reversed(range(0, n - 1)):
        x[i] = (b[i] - U[i, i + 1:].dot(x[i + 1:]).item()) / U[i, i]
    
    return x


def determinant(p: list[int], L: np.ndarray, U: np.ndarray = None) -> float:
    if U is None:
        U = L
    res = 1
    n = len(p)
    for i in range(n):
        res *= U[i, i]
    res *= (-1) ** count_permutation(p)
    return res


def solve_system(p: list[int], L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = forward_pass(L, b[p])
    x = backward_pass(U, z)
    return x


def inverse_matrix(p: list[int], L: np.ndarray, U: np.ndarray, epsilon: float = 1e-8):
    det = determinant(p, U)
    if abs(det) < epsilon:
        raise ValueError(f'Матрица вырождена, определитель равен: {det:.6g}')
    
    n = len(p)
    X = np.zeros((n, n))
    I = np.eye(n)

    for i in range(n):
        x_i = solve_system(p, L, U, I[:, i])
        X[:, i] = x_i
    
    return X
