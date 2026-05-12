from collections.abc import Callable

import numpy as np


SystemFunction = Callable[[float, np.ndarray], np.ndarray]
SecondOrderRightSide = Callable[[float, float, float], float]


def _build_uniform_grid(left: float, right: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError('Шаг должен быть положительным')
    if right <= left:
        raise ValueError('Правая граница должна быть больше левой')

    interval = right - left
    intervals_count = int(round(interval / step))
    if intervals_count <= 0 or not np.isclose(intervals_count * step, interval):
        raise ValueError('Шаг должен нацело разбивать отрезок')

    return np.linspace(left, right, intervals_count + 1)


def _as_state(initial_state: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    state = np.asarray(initial_state, dtype=float)
    if state.ndim != 1:
        raise ValueError('Начальное условие должно быть одномерным вектором')
    return state


def reduce_second_order_ode(right_side: SecondOrderRightSide) -> SystemFunction:
    def system(x: float, state: np.ndarray) -> np.ndarray:
        y, derivative = state
        return np.array([
            derivative,
            right_side(x, y, derivative),
        ], dtype=float)

    return system


def euler_method(
    function: SystemFunction,
    left: float,
    right: float,
    step: float,
    initial_state: np.ndarray | list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    x = _build_uniform_grid(left, right, step)
    y = np.zeros((x.size, _as_state(initial_state).size), dtype=float)
    y[0] = _as_state(initial_state)

    for i in range(x.size - 1):
        y[i + 1] = y[i] + step * function(x[i], y[i])

    return x, y


def runge_kutta_method(
    function: SystemFunction,
    left: float,
    right: float,
    step: float,
    initial_state: np.ndarray | list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    x = _build_uniform_grid(left, right, step)
    y = np.zeros((x.size, _as_state(initial_state).size), dtype=float)
    y[0] = _as_state(initial_state)

    for i in range(x.size - 1):
        k1 = step * function(x[i], y[i])
        k2 = step * function(x[i] + step / 2, y[i] + k1 / 2)
        k3 = step * function(x[i] + step / 2, y[i] + k2 / 2)
        k4 = step * function(x[i] + step, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x, y


def adams_bashforth_method(
    function: SystemFunction,
    left: float,
    right: float,
    step: float,
    initial_state: np.ndarray | list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    x = _build_uniform_grid(left, right, step)
    if x.size < 4:
        raise ValueError('Методу Адамса 4-го порядка нужно минимум 4 узла')

    y = np.zeros((x.size, _as_state(initial_state).size), dtype=float)
    _, starter = runge_kutta_method(function, left, left + 3 * step, step, initial_state)
    y[:4] = starter

    for i in range(3, x.size - 1):
        f_i = function(x[i], y[i])
        f_i_1 = function(x[i - 1], y[i - 1])
        f_i_2 = function(x[i - 2], y[i - 2])
        f_i_3 = function(x[i - 3], y[i - 3])
        y[i + 1] = y[i] + step / 24 * (55 * f_i - 59 * f_i_1 + 37 * f_i_2 - 9 * f_i_3)

    return x, y


def runge_romberg_errors(
    coarse_values: np.ndarray,
    fine_values: np.ndarray,
    coarse_step: float,
    fine_step: float,
    order: int,
) -> np.ndarray:
    if coarse_step <= fine_step:
        raise ValueError('coarse_step должен быть больше fine_step')
    if order <= 0:
        raise ValueError('Порядок метода должен быть положительным')

    ratio = coarse_step / fine_step
    rounded_ratio = int(round(ratio))
    if not np.isclose(ratio, rounded_ratio):
        raise ValueError('Отношение шагов должно быть целым')

    fine_on_coarse_grid = np.asarray(fine_values)[::rounded_ratio]
    coarse_values = np.asarray(coarse_values)
    if coarse_values.shape != fine_on_coarse_grid.shape:
        raise ValueError('Сетки с разными шагами должны описывать один отрезок')

    return np.abs((fine_on_coarse_grid - coarse_values) / (ratio ** order - 1))


def exact_solution_error(
    exact_solution: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    values: np.ndarray,
    component: int = 0,
) -> np.ndarray:
    return np.abs(np.asarray(exact_solution(x)) - np.asarray(values)[:, component])
