import sys
import math
import numpy as np
from scipy.sparse import csr_matrix
from ..utils import csr_row_norm


def is_diagonally_dominant(A: csr_matrix):
    diagonals = np.abs(A.diagonal())
    rows_sum = np.add.reduceat(
        np.abs(A.data),
        A.indptr[:-1]
    ) -  diagonals
    return np.all(diagonals > rows_sum)


def jacobi_preprocessing(
    A: csr_matrix,
    b: np.ndarray,
    machine_epsilon: float = 1e-8
):
    n = A.shape[0]
    Al = A.astype(np.float64).tolil().copy()
    for i in range(n - 1):
        if np.abs(Al[i, i]) < machine_epsilon:
            j = i + 1 + np.argmax(np.abs(Al[i + 1:, i]))
            Al[[i, j]] = Al[[j, i]]
    A = Al.tocsr()

    is_diag = True
    if not is_diagonally_dominant(A):
        is_diag = False
        sys.stderr.write('[WARNING] Jacobi Preprocessing: Norm of matrix A is greater than 1, convergence is not guaranteed.\n')

    diagonal = A.diagonal()
    for i in range(A.shape[0]):
        start = A.indptr[i]
        end = A.indptr[i+1]
        A.data[start:end] /= -diagonal[i]
    b = b.squeeze() / diagonal

    for i in range(n):
        A[i, i] = 0
    return A, b, is_diag


def simple_iterations(
    A: csr_matrix,
    b: np.ndarray,
    precision_epsilon: float = 1e-4,
    machine_epsilon: float = 1e-8
):
    A, b, is_diag = jacobi_preprocessing(A, b, machine_epsilon)

    alpha = csr_row_norm(A)
    beta = np.max(np.abs(b))
    k_aprior = math.ceil((np.log(precision_epsilon) - np.log(beta) + np.log(1 - alpha)) / np.log(alpha) - 1)
    sys.stderr.write(f'[INFO] Method of Simple Iterations: a priori iteration count estimate is {k_aprior}\n')

    x_prev = b
    for k in range(1, k_aprior + 1):
        x_cur = b + A.dot(x_prev)
        factor = alpha / (1 - alpha) if is_diag else 1
        epsilon_k = np.max(np.abs(x_cur - x_prev)) * factor
        x_prev = x_cur
        if epsilon_k <= precision_epsilon:
            sys.stderr.write(f'[INFO] Method of Simple Iterations: a posteriori iteration count is {k}\n')
            break
    return x_prev


def seidel_method(
    A: csr_matrix,
    b: np.ndarray,
    precision_epsilon: float = 1e-4,
    machine_epsilon: float = 1e-8
):
    n = A.shape[0]
    A, b, is_diag = jacobi_preprocessing(A, b, machine_epsilon)

    alpha = csr_row_norm(A)
    beta = np.max(np.abs(b))
    k_aprior = math.ceil((np.log(precision_epsilon) - np.log(beta) + np.log(1 - alpha)) / np.log(alpha) - 1)
    sys.stderr.write(f'[INFO] Method of Simple Iterations: a priori iteration count estimate is {k_aprior}\n')

    x_prev = b
    for k in range(1, k_aprior + 1):
        x_cur = x_prev.copy()
        for i in range(n):
            x_cur[i] = b[i] + A[i].dot(x_cur)

        factor = alpha / (1 - alpha) if is_diag else 1
        epsilon_k = np.max(np.abs(x_cur - x_prev)) * factor
        x_prev = x_cur
        if epsilon_k <= precision_epsilon:
            sys.stderr.write(f'[INFO] Method of Simple Iterations: a posteriori iteration count is {k}\n')
            break
    return x_prev
