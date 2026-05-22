from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..nonlinear import secant
from .ivp import SecondOrderRightSide, SystemFunction, reduce_second_order_ode


LinearCoefficient = Callable[[float], float] | float


@dataclass(frozen=True)
class BoundaryCondition:
    alpha: float
    beta: float
    value: float

    def residual(self, y: float, derivative: float) -> float:
        return self.alpha * y + self.beta * derivative - self.value

    def state_from_value(self, y: float) -> np.ndarray:
        if np.isclose(self.beta, 0.0):
            raise ValueError('Нельзя восстановить производную: beta равен нулю')
        return np.array([y, (self.value - self.alpha * y) / self.beta], dtype=float)

    def state_from_derivative(self, derivative: float) -> np.ndarray:
        if np.isclose(self.alpha, 0.0):
            raise ValueError('Нельзя восстановить значение функции: alpha равен нулю')
        return np.array([(self.value - self.beta * derivative) / self.alpha, derivative], dtype=float)


def _build_directional_grid(start: float, finish: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError('Шаг должен быть положительным')
    if np.isclose(start, finish):
        raise ValueError('Границы отрезка должны различаться')

    interval = abs(finish - start)
    intervals_count = int(round(interval / step))
    if intervals_count <= 0 or not np.isclose(intervals_count * step, interval):
        raise ValueError('Шаг должен нацело разбивать отрезок')

    return np.linspace(start, finish, intervals_count + 1)


def _runge_kutta_directional(
    function: SystemFunction,
    start: float,
    finish: float,
    step: float,
    initial_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = _build_directional_grid(start, finish, step)
    y = np.zeros((x.size, initial_state.size), dtype=float)
    y[0] = initial_state

    for i in range(x.size - 1):
        h = x[i + 1] - x[i]
        k1 = h * function(x[i], y[i])
        k2 = h * function(x[i] + h / 2, y[i] + k1 / 2)
        k3 = h * function(x[i] + h / 2, y[i] + k2 / 2)
        k4 = h * function(x[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x, y


def shooting_method(
    right_side: SecondOrderRightSide,
    left: float,
    right: float,
    step: float,
    left_boundary: BoundaryCondition,
    right_boundary: BoundaryCondition,
    eta0: float,
    eta1: float,
    precision_epsilon: float = 1e-6,
    max_iter: int = 1000,
    shoot_from: str = 'left',
    unknown: str = 'auto',
) -> tuple[np.ndarray, np.ndarray, float, Callable[[float], float]]:
    system = reduce_second_order_ode(right_side)

    if shoot_from not in {'left', 'right'}:
        raise ValueError("shoot_from должен быть 'left' или 'right'")
    if unknown not in {'auto', 'value', 'derivative'}:
        raise ValueError("unknown должен быть 'auto', 'value' или 'derivative'")

    start_boundary = left_boundary if shoot_from == 'left' else right_boundary
    finish_boundary = right_boundary if shoot_from == 'left' else left_boundary
    start = left if shoot_from == 'left' else right
    finish = right if shoot_from == 'left' else left

    if unknown == 'auto':
        unknown = 'value' if not np.isclose(start_boundary.beta, 0.0) else 'derivative'

    def initial_state(eta: float) -> np.ndarray:
        if unknown == 'value':
            return start_boundary.state_from_value(eta)
        return start_boundary.state_from_derivative(eta)

    def residual(eta: float) -> float:
        _, values = _runge_kutta_directional(system, start, finish, step, initial_state(eta))
        return finish_boundary.residual(values[-1, 0], values[-1, 1])

    eta = secant(
        residual,
        x0=eta0,
        x1=eta1,
        precision_epsilon=precision_epsilon,
        max_iter=max_iter,
    )
    x, values = _runge_kutta_directional(system, start, finish, step, initial_state(eta))

    if shoot_from == 'right':
        x = x[::-1]
        values = values[::-1]

    return x, values, eta, residual


def _coefficient_value(coefficient: LinearCoefficient, x: float) -> float:
    if callable(coefficient):
        return float(coefficient(x))
    return float(coefficient)


def finite_difference_method(
    b: LinearCoefficient,
    c: LinearCoefficient,
    f: LinearCoefficient,
    left: float,
    right: float,
    step: float,
    left_boundary: BoundaryCondition,
    right_boundary: BoundaryCondition,
    *,
    a: LinearCoefficient = 1.0,
    boundary_order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    if boundary_order not in {1, 2}:
        raise ValueError('Порядок аппроксимации граничных условий должен быть 1 или 2')

    x = _build_directional_grid(left, right, step)
    size = x.size
    if size < 3:
        raise ValueError('Нужно минимум три узла сетки')

    matrix = np.zeros((size, size), dtype=float)
    rhs = np.zeros(size, dtype=float)

    for i in range(1, size - 1):
        x_i = float(x[i])
        a_i = _coefficient_value(a, x_i)
        b_i = _coefficient_value(b, x_i)
        c_i = _coefficient_value(c, x_i)
        f_i = _coefficient_value(f, x_i)
        matrix[i, i - 1] = a_i / step ** 2 - b_i / (2 * step)
        matrix[i, i] = -2 * a_i / step ** 2 + c_i
        matrix[i, i + 1] = a_i / step ** 2 + b_i / (2 * step)
        rhs[i] = f_i

    if boundary_order == 1:
        matrix[0, 0] = left_boundary.alpha - left_boundary.beta / step
        matrix[0, 1] = left_boundary.beta / step
        rhs[0] = left_boundary.value

        matrix[-1, -2] = -right_boundary.beta / step
        matrix[-1, -1] = right_boundary.alpha + right_boundary.beta / step
        rhs[-1] = right_boundary.value
    else:
        matrix[0, 0] = left_boundary.alpha - 3 * left_boundary.beta / (2 * step)
        matrix[0, 1] = 2 * left_boundary.beta / step
        matrix[0, 2] = -left_boundary.beta / (2 * step)
        rhs[0] = left_boundary.value

        matrix[-1, -3] = right_boundary.beta / (2 * step)
        matrix[-1, -2] = -2 * right_boundary.beta / step
        matrix[-1, -1] = right_boundary.alpha + 3 * right_boundary.beta / (2 * step)
        rhs[-1] = right_boundary.value

    return x, np.linalg.solve(matrix, rhs)


def runge_romberg_grid_errors(
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
