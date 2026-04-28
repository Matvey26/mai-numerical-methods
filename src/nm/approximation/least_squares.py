import numpy as np

from ..linalg import lu_decomposition, solve_system


def _validate_points(
    x_nodes: list[float] | np.ndarray,
    y_nodes: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('Точки должны быть одномерными массивами')
    if x.size != y.size:
        raise ValueError('Количество x и y должно совпадать')
    if x.size == 0:
        raise ValueError('Нужна хотя бы одна точка')

    return x, y


def least_squares_normal_system(
    x_nodes: list[float] | np.ndarray,
    y_nodes: list[float] | np.ndarray,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    if degree < 0:
        raise ValueError('Степень многочлена должна быть неотрицательной')

    x, y = _validate_points(x_nodes, y_nodes)
    feature_matrix = np.vander(x, N=degree + 1, increasing=True)

    return feature_matrix.T.dot(feature_matrix), feature_matrix.T.dot(y)


def least_squares_polynomial(
    x_nodes: list[float] | np.ndarray,
    y_nodes: list[float] | np.ndarray,
    degree: int,
) -> np.poly1d:
    normal_matrix, right_side = least_squares_normal_system(x_nodes, y_nodes, degree)
    p, L, U = lu_decomposition(normal_matrix)
    coefficients = solve_system(p, L, U, right_side)

    return np.poly1d(coefficients[::-1])


def squared_error_sum(
    polynomial: np.poly1d,
    x_nodes: list[float] | np.ndarray,
    y_nodes: list[float] | np.ndarray,
) -> float:
    x, y = _validate_points(x_nodes, y_nodes)
    residuals = polynomial(x) - y

    return residuals.dot(residuals).item()
