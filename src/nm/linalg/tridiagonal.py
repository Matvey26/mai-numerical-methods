"""
https://moodle.kstu.ru/pluginfile.php/512336/mod_resource/content/1/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4%20%D0%BF%D1%80%D0%BE%D0%B3%D0%BE%D0%BD%D0%BA%D0%B8.pdf
"""
from ..utils import Matrix


def is_diagonally_dominant(A: list[tuple]) -> bool:
    # strong = False
    n = len(A)
    for i in range(1, n - 1):
        a = A[i][0] if i > 0 else 0
        b = A[i][1]
        c = A[i][2] if i < n - 1 else 0
        if abs(b) <= abs(a) + abs(c):
            return False
        # if abs(b) > abs(a) + abs(c):
        #     strong = True

    return True


def thomas_algorithm(A: list[tuple], b: list[float]) -> Matrix:
    """
    A = [(a1, b1, c1), (a2, b2, c2), ..., (an, bn, cn)],
    where a1 = 0, cn = 0.
    """
    if not is_diagonally_dominant(A):
        raise ValueError('Для использования метода прогонки матрица '
                         'A должна иметь диагональное преобладание')

    n = len(A)
    alpha = [0] * n
    beta = [0] * n
    alpha[0] = beta[0] = None  # эти значения не используются

    alpha[1] = -A[0][2] / A[0][1]
    beta[1] = b[0] / A[0][1]

    for i in range(2, n):
        znam = A[i - 1][0] * alpha[i - 1] + A[i - 1][1]
        alpha[i] = -A[i - 1][2] / znam
        beta[i] = (b[i - 1] - A[i - 1][0] * beta[i - 1]) / znam

    x = [0] * n
    x[n - 1] = (b[n - 1] - A[n - 1][0] * beta[n - 1]) / (A[n - 1][0] * alpha[n - 1] + A[n - 1][1])
    for i in reversed(range(n - 1)):
        x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]

    return x
