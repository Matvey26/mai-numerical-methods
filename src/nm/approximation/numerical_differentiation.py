import numpy as np


def first_derivative_first_order(
    x_nodes: list[float] | np.ndarray,
    y_nodes: list[float] | np.ndarray,
    x_value: float,
    direction: str = 'center',
) -> float:
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('Узлы должны быть одномерными массивами')
    if x.size != y.size:
        raise ValueError('Количество узлов x и значений y должно совпадать')
    if x.size < 2:
        raise ValueError('Для вычисления производной нужно минимум два узла')

    if direction == 'left':
        i = np.searchsorted(x, x_value, side='right') - 2
        i = max(i, 0)
    elif direction == 'right':
        i = np.searchsorted(x, x_value, side='left')
        if i >= x.size - 1:
            i = x.size - 2
    elif direction == 'center':
        i = np.searchsorted(x, x_value, side='right') - 1
        i = min(max(i, 0), x.size - 2)
        if i + 1 < x.size and abs(x[i + 1] - x_value) < abs(x_value - x[i]):
            i = min(i + 1, x.size - 2)
    else:
        raise ValueError("direction должен быть 'left', 'right' или 'center'")

    return ((y[i + 1] - y[i]) / (x[i + 1] - x[i])).item()


def first_derivative_second_order(
    x_nodes: list[float] | np.ndarray,
    y_nodes: list[float] | np.ndarray,
    x_value: float,
) -> float:
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('Узлы должны быть одномерными массивами')
    if x.size != y.size:
        raise ValueError('Количество узлов x и значений y должно совпадать')
    if x.size < 3:
        raise ValueError('Для вычисления производной нужно минимум три узла')

    idx = np.searchsorted(x, x_value, side='right')
    idx = min(max(idx, 1), x.size - 2)

    i = idx - 1

    h1 = x[i + 1] - x[i]
    h2 = x[i + 2] - x[i + 1]

    f1 = (y[i + 1] - y[i]) / h1
    f2 = (y[i + 2] - y[i + 1]) / h2

    return (
        f1 + (f2 - f1) / (x[i + 2] - x[i]) * (2 * x_value - x[i] - x[i + 1])
    ).item()


def second_derivative(
    x_nodes: list[float] | np.ndarray,
    y_nodes: list[float] | np.ndarray,
    x_value: float,
) -> float:
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('Узлы должны быть одномерными массивами')
    if x.size != y.size:
        raise ValueError('Количество узлов x и значений y должно совпадать')
    if x.size < 3:
        raise ValueError('Для вычисления второй производной нужно минимум три узла')

    idx = np.searchsorted(x, x_value, side='right')
    idx = min(max(idx, 1), x.size - 2)

    i = idx - 1

    h1 = x[i + 1] - x[i]
    h2 = x[i + 2] - x[i + 1]

    f1 = (y[i + 1] - y[i]) / h1
    f2 = (y[i + 2] - y[i + 1]) / h2

    return (2 * (f2 - f1) / (x[i + 2] - x[i])).item()
