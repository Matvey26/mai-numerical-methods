import numpy as np


def _validate_nodes(
    x_nodes: list[float] | np.ndarray,
    y_nodes: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('Узлы интерполяции должны быть одномерными массивами')
    if x.size != y.size:
        raise ValueError('Количество узлов x и значений y должно совпадать')
    if x.size == 0:
        raise ValueError('Нужен хотя бы один узел интерполяции')
    if np.unique(x).size != x.size:
        raise ValueError('Узлы интерполяции должны быть различными')

    return x, y


def lagrange_polynomial(
    x_nodes: list[float] | np.ndarray,
    y_nodes: list[float] | np.ndarray,
) -> np.poly1d:
    x, y = _validate_nodes(x_nodes, y_nodes)

    polynomial = np.poly1d([0.0])
    for i, x_i in enumerate(x):
        basis = np.poly1d([1.0])
        denominator = 1.0

        for j, x_j in enumerate(x):
            if i == j:
                continue
            basis *= np.poly1d([1.0, -x_j])
            denominator *= x_i - x_j

        polynomial += y[i] * basis / denominator

    return polynomial


def divided_differences(
    x_nodes: list[float] | np.ndarray,
    y_nodes: list[float] | np.ndarray,
) -> np.ndarray:
    x, y = _validate_nodes(x_nodes, y_nodes)

    coefficients = y.astype(float).copy()
    n = x.size
    for order in range(1, n):
        coefficients[order:n] = (
            (coefficients[order:n] - coefficients[order - 1:n - 1])
            / (x[order:n] - x[:n - order])
        )

    return coefficients


def newton_polynomial(
    x_nodes: list[float] | np.ndarray,
    y_nodes: list[float] | np.ndarray,
) -> np.poly1d:
    x, y = _validate_nodes(x_nodes, y_nodes)
    coefficients = divided_differences(x, y)

    polynomial = np.poly1d([coefficients[-1]])
    for i in reversed(range(x.size - 1)):
        polynomial = polynomial * np.poly1d([1.0, -x[i]]) + coefficients[i]

    return polynomial
