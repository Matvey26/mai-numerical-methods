import numpy as np


def _validate_spline_nodes(
    x_nodes: list[float] | np.ndarray,
    y_nodes: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('Узлы сплайна должны быть одномерными массивами')
    if x.size != y.size:
        raise ValueError('Количество узлов x и значений y должно совпадать')
    if x.size < 2:
        raise ValueError('Для построения сплайна нужно минимум два узла')
    if not np.all(np.diff(x) > 0):
        raise ValueError('Узлы сплайна должны быть строго возрастающими')

    return x, y


def natural_cubic_spline_coefficients(
    x_nodes: list[float] | np.ndarray,
    y_nodes: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = _validate_spline_nodes(x_nodes, y_nodes)
    n = x.size
    h = np.diff(x)

    system = np.zeros((n, n), dtype=float)
    right_side = np.zeros(n, dtype=float)
    system[0, 0] = 1.0
    system[-1, -1] = 1.0

    for i in range(1, n - 1):
        system[i, i - 1] = h[i - 1]
        system[i, i] = 2 * (h[i - 1] + h[i])
        system[i, i + 1] = h[i]
        right_side[i] = 6 * (
            (y[i + 1] - y[i]) / h[i]
            - (y[i] - y[i - 1]) / h[i - 1]
        )

    second_derivatives = np.linalg.solve(system, right_side)

    a = y[:-1].copy()
    b = (
        np.diff(y) / h
        - h * (2 * second_derivatives[:-1] + second_derivatives[1:]) / 6
    )
    c = second_derivatives[:-1] / 2
    d = np.diff(second_derivatives) / (6 * h)

    return a, b, c, d


def evaluate_cubic_spline(
    x_nodes: list[float] | np.ndarray,
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    x_value: float,
) -> float:
    x = np.asarray(x_nodes, dtype=float)
    a, b, c, d = coefficients

    if x.ndim != 1:
        raise ValueError('Узлы сплайна должны быть одномерным массивом')
    if not np.all(np.diff(x) > 0):
        raise ValueError('Узлы сплайна должны быть строго возрастающими')
    if any(np.asarray(part).size != x.size - 1 for part in coefficients):
        raise ValueError('Коэффициенты сплайна должны быть заданы для каждого интервала')
    if x_value < x[0] or x_value > x[-1]:
        raise ValueError('Точка вычисления должна лежать внутри отрезка интерполяции')

    interval_idx = np.searchsorted(x, x_value, side='right') - 1
    interval_idx = min(max(interval_idx, 0), x.size - 2)
    dx = x_value - x[interval_idx]

    return (
        a[interval_idx]
        + b[interval_idx] * dx
        + c[interval_idx] * dx ** 2
        + d[interval_idx] * dx ** 3
    ).item()
