from collections.abc import Callable

import numpy as np


def _build_uniform_grid(left: float, right: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError('Шаг интегрирования должен быть положительным')
    if right <= left:
        raise ValueError('Правая граница должна быть больше левой')

    interval = right - left
    intervals_count = int(round(interval / step))
    if intervals_count <= 0 or not np.isclose(intervals_count * step, interval):
        raise ValueError('Шаг должен нацело разбивать отрезок интегрирования')

    return np.linspace(left, right, intervals_count + 1)


def rectangle_rule(
    function: Callable[[np.ndarray], np.ndarray],
    left: float,
    right: float,
    step: float,
) -> float:
    x = _build_uniform_grid(left, right, step)
    middle = (x[:-1] + x[1:]) / 2

    return (step * np.sum(function(middle))).item()


def trapezoidal_rule(
    function: Callable[[np.ndarray], np.ndarray],
    left: float,
    right: float,
    step: float,
) -> float:
    x = _build_uniform_grid(left, right, step)
    y = function(x)

    return (step * (np.sum(y[1:-1]) + (y[0] + y[-1]) / 2)).item()


def simpson_rule(
    function: Callable[[np.ndarray], np.ndarray],
    left: float,
    right: float,
    step: float,
) -> float:
    x = _build_uniform_grid(left, right, step)
    intervals_count = x.size - 1
    if intervals_count % 2 != 0:
        raise ValueError('Формула Симпсона требует четное количество отрезков')

    y = function(x)

    return (
        step
        / 3
        * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))
    ).item()


def runge_romberg_error(
    coarse_value: float,
    fine_value: float,
    coarse_step: float,
    fine_step: float,
    order: int,
) -> float:
    if coarse_step <= fine_step:
        raise ValueError('coarse_step должен быть больше fine_step')
    if order <= 0:
        raise ValueError('Порядок метода должен быть положительным')

    ratio = coarse_step / fine_step

    return (fine_value - coarse_value) / (ratio ** order - 1)


def rectangle_error_bound(
    left: float,
    right: float,
    step: float,
    second_derivative_bound: float,
) -> float:
    return step ** 2 * second_derivative_bound * (right - left) / 24


def trapezoidal_error_bound(
    left: float,
    right: float,
    step: float,
    second_derivative_bound: float,
) -> float:
    return (right - left) * step ** 2 * second_derivative_bound / 12


def simpson_error_bound(
    left: float,
    right: float,
    step: float,
    fourth_derivative_bound: float,
) -> float:
    return (right - left) * step ** 4 * fourth_derivative_bound / 180
