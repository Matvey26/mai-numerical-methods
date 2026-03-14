import numpy as np


def _find_max_higher_diagonal(A: np.ndarray):
    upper = np.triu(A, k=1)
    idx = np.argmax(np.abs(upper)).item()
    i, j = idx // A.shape[1], idx % A.shape[1]
    return i, j


def _calculate_phi(A: np.ndarray, index: tuple[float, float]):
    i, j = index
    if np.allclose(A[i, i], A[j, j]):
        return np.pi / 4
    phi = 0.5 * np.arctan(2 * A[i, j] / (A[i, i] - A[j, j]))
    return phi


def _stop_criteria(A: np.ndarray, epsilon: float):
    upper = np.triu(A, k=1)
    return np.sqrt(np.sum(upper ** 2)) <= epsilon


def _rotate(A: np.ndarray, phi: np.number, index: tuple[float, float]) -> np.ndarray:
    A = A.astype(np.float64)
    i, j = index

    new_i = A[:, i] * np.cos(phi) + A[:, j] * np.sin(phi)
    new_j = -A[:, i] * np.sin(phi) + A[:, j] * np.cos(phi)
    A[:, i] = new_i
    A[:, j] = new_j
    return A


def eigen_jacobi(A: np.ndarray, epsilon: float = 1e-3, n_iter: int = None):
    if not np.allclose(A, A.T):
        raise ValueError('Для корректного использования метода '
                         'Якоби матрица должна быть симметрической')

    A = A.astype(np.float64).copy()
    n = A.shape[0]
    eigen_vectors = np.eye(n)

    iterations_count = 0
    stop = (
        (lambda: _stop_criteria(A, epsilon))
        if n_iter is None
        else (lambda: iterations_count >= n_iter)
    )
    while not stop():
        index = _find_max_higher_diagonal(A)
        phi = _calculate_phi(A, index)
        A = _rotate(_rotate(A, phi, index).T, phi, index).T
        eigen_vectors = _rotate(eigen_vectors, phi, index)
        iterations_count += 1
    
    return A.diagonal(), eigen_vectors, iterations_count


if __name__ == '__main__':
    A = np.array([
        [4, 2, 1],
        [2, 5, 3],
        [1, 3, 6],
    ])

    val, vec, cnt = eigen_jacobi(A, n_iter=3)
    print(val)
    print(vec)
    print(cnt)
