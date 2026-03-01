"""
https://stepanzh.github.io/computational_thermodynamics/syslinear/plu.html
"""
from ..utils import Matrix, count_permutation


def lu_decomposition(A: Matrix, epsilon: float = 1e-8) -> tuple[list[int], Matrix, Matrix]:

    if A.shape[0] != A.shape[1]:
        raise ValueError('Матрица A должна быть квадратной')
    
    n = A.shape[0]

    U = Matrix.zeros(shape=A.shape)
    L = Matrix.zeros(shape=A.shape)
    p = [0] * n

    A_k = A.copy()

    for k in range(n - 1):
        p[k] = max(range(n), key=lambda i: abs(A_k[i, k]))

        if abs(A_k[p[k], k]) < epsilon:
            raise ValueError('Матрица A плохо обусловлена (или вырождена)')

        U[k, k:] = A_k[p[k], k:]
        L[:, k] = A_k[:, k] / U[k, k]
        A_k -= L[:, k].dot(U[k, :])

    p[n - 1] = max(range(n), key=lambda i: abs(A_k[i, n - 1]))
    U[n - 1, n - 1] = A_k[p[n - 1], n - 1]
    L[:, n - 1] = A_k[:, n - 1] / U[n - 1, n - 1]

    return p, L[p, :], U


def forward_pass(L: Matrix, b: Matrix) -> Matrix:
    n = b.shape[0]
    x = Matrix.zeros(shape=(n, 1))
    x[0, 0] = b[0, 0] / L[0, 0]
    for i in range(1, n):
        x[i, 0] = (b[i, 0] - L[i, :i].dot(x[:i, 0])[0, 0]) / L[i, i]
    
    return x


def backward_pass(U: Matrix, b: Matrix) -> Matrix:
    n = b.shape[0]
    x = Matrix.zeros(shape=(n, 1))
    x[n - 1, 0] = b[n - 1, 0] / U[n - 1, n - 1]
    for i in reversed(range(0, n - 1)):
        x[i, 0] = (b[i, 0] - U[i, i + 1:].dot(x[i + 1:, 0])[0, 0]) / U[i, i]
    
    return x


def determinant(p: list[int], L: Matrix, U: Matrix = None) -> float:
    if U is None:
        U = L
    res = 1
    n = len(p)
    for i in range(n):
        res *= U[i, i]
    res *= (-1) ** count_permutation(p)
    return res


def solve_system(p: list[int], L: Matrix, U: Matrix, b: Matrix) -> Matrix:
    z = forward_pass(L, b[p])
    x = backward_pass(U, z)
    return x


def inverse_matrix(p: list[int], L: Matrix, U: Matrix, epsilon: float = 1e-8):
    det = determinant(p, U)
    if abs(det) < epsilon:
        raise ValueError(f'Матрица вырождена, определитель равен: {det:.6g}')
    
    n = len(p)
    X = Matrix.zeros(shape=(n, n))
    I = Matrix.eye(size=n)

    for i in range(n):
        x_i = solve_system(p, L, U, I[:, i])
        X[:, i] = x_i
    
    return X
