from collections.abc import Callable
from math import factorial

import numpy as np


def _validate_x_nodes(x_nodes: list[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(x_nodes, dtype=float)

    if x.ndim != 1:
        raise ValueError('Узлы интерполяции должны быть одномерным массивом')
    if x.size == 0:
        raise ValueError('Нужен хотя бы один узел интерполяции')
    if np.unique(x).size != x.size:
        raise ValueError('Узлы интерполяции должны быть различными')

    return x


def _validate_nodes(
    x_nodes: list[float] | np.ndarray,
    y_nodes: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = _validate_x_nodes(x_nodes)
    y = np.asarray(y_nodes, dtype=float)

    if y.ndim != 1:
        raise ValueError('Узлы интерполяции должны быть одномерными массивами')
    if x.size != y.size:
        raise ValueError('Количество узлов x и значений y должно совпадать')

    return x, y


def interpolation_omega(x_nodes: list[float] | np.ndarray) -> np.poly1d:
    x = _validate_x_nodes(x_nodes)

    omega = np.poly1d([1.0])
    for x_i in x:
        omega *= np.poly1d([1.0, -x_i])

    return omega


def lagrange_polynomial(
    x_nodes: list[float] | np.ndarray,
    y_nodes: list[float] | np.ndarray,
) -> np.poly1d:
    x, y = _validate_nodes(x_nodes, y_nodes)
    n = len(x)

    omega = interpolation_omega(x)
    omega_deriv = omega.deriv()
    omega_deriv_at_nodes = [omega_deriv(x_i) for x_i in x]

    polynomial = np.poly1d([0.0])
    for i in range(n):
        basis, remainder = np.polydiv(omega, np.poly1d([1.0, -x[i]]))
        if not np.allclose(remainder, 0.0):
            raise RuntimeError('Не удалось разделить omega(x) на базисный множитель')
        polynomial += y[i] * basis / omega_deriv_at_nodes[i]

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


def interpolation_actual_error(
    func: Callable[[float], float],
    polynomial: np.poly1d,
    x_value: float,
) -> float:
    return abs(func(x_value) - polynomial(x_value)).item()


def interpolation_apriori_error_bound(
    x_nodes: list[float] | np.ndarray,
    x_value: float,
    derivative: Callable,
    samples_count: int = 10_000,
) -> float:
    x = _validate_x_nodes(x_nodes)
    if samples_count < 2:
        raise ValueError('Для оценки максимума нужно минимум две точки сетки')

    grid = np.linspace(np.min(x), np.max(x), samples_count)
    try:
        derivative_values = np.asarray(derivative(grid), dtype=float)
    except Exception:
        derivative_values = np.array([derivative(point) for point in grid], dtype=float)

    if derivative_values.shape != grid.shape:
        derivative_values = np.array([derivative(point) for point in grid], dtype=float)

    max_derivative = np.max(np.abs(derivative_values))
    omega_value = abs(interpolation_omega(x)(x_value))

    return (max_derivative * omega_value / factorial(x.size)).item()


def interpolation_aposteriori_error_estimate(
    x_nodes: list[float] | np.ndarray,
    y_nodes: list[float] | np.ndarray,
    x_value: float,
    extra_x: float,
    extra_y: float,
) -> float:
    x, y = _validate_nodes(x_nodes, y_nodes)
    if np.any(np.isclose(x, extra_x)):
        raise ValueError('Дополнительный узел уже входит в таблицу интерполяции')

    augmented_x = np.append(x, extra_x)
    augmented_y = np.append(y, extra_y)
    first_omitted_coefficient = divided_differences(augmented_x, augmented_y)[-1]
    omega_value = interpolation_omega(x)(x_value)

    return abs(first_omitted_coefficient * omega_value).item()
