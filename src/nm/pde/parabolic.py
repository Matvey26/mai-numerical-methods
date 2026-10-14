"""Лаба 5: u_t = a*u_xx + b*u_x + c*u + f(x, t), a > 0.

Коэффициенты a, b, c постоянны. На концах отрезка задаются условия
alpha*u + beta*u_x = g(t). Производная направлена вдоль оси x на ОБОИХ
концах (это не производная по внешней нормали).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BoundaryCondition:
    alpha: float
    beta: float
    value: Callable[[float], float]


def tridiagonal_solve(lower, diagonal, upper, rhs):
    """Прогонка: lower[0] = upper[-1] = 0; входные массивы не меняются."""
    diagonal, rhs = diagonal.copy(), rhs.copy()
    # Достаточное условие устойчивой прогонки — диагональное преобладание.
    # Но оно не необходимо: строки условий Робина могут его не иметь.
    # Поэтому не запрещаем такие системы, а проверяем реальные знаменатели.
    tolerance = 32 * np.finfo(float).eps * max(
        np.max(np.abs(diagonal)), np.max(np.abs(lower)),
        np.max(np.abs(upper)), 1.0,
    )
    for i in range(1, len(rhs)):
        if abs(diagonal[i - 1]) <= tolerance:
            raise ArithmeticError("Нулевой знаменатель прогонки; измените сетку.")
        multiplier = lower[i] / diagonal[i - 1]
        diagonal[i] -= multiplier * upper[i - 1]
        rhs[i] -= multiplier * rhs[i - 1]
    if abs(diagonal[-1]) <= tolerance:
        raise ArithmeticError("Нулевой последний знаменатель прогонки.")
    result = np.empty_like(rhs)
    result[-1] = rhs[-1] / diagonal[-1]
    for i in range(len(rhs) - 2, -1, -1):
        result[i] = (rhs[i] - upper[i] * result[i + 1]) / diagonal[i]
    return result


def solve_parabolic(
    a, b, c, source, initial, left_boundary, right_boundary,
    left, right, final_time, nx, nt, theta=0.5, boundary_order="2p2",
):
    """Вернуть x, t, U, где U[k, j] приближает u(x[j], t[k]).

    nx и nt — число интервалов, а не узлов. theta=0: явная схема;
    theta=1: неявная; theta=0.5: Кранк–Николсон.
    boundary_order: '2p1', '3p2', '2p2' — две точки/первый порядок,
    три точки/второй, две точки/второй с использованием уравнения.
    initial и source должны поддерживать массив x (обычные функции NumPy).
    """
    if a <= 0 or right <= left or final_time <= 0 or nx < 3 or nt < 1:
        raise ValueError("Нужны a > 0, корректный отрезок, T > 0, nx >= 3, nt >= 1.")
    if theta not in (0, 0.5, 1) or boundary_order not in ("2p1", "3p2", "2p2"):
        raise ValueError("Неизвестная схема или аппроксимация границ.")
    x = np.linspace(left, right, nx + 1)
    t = np.linspace(0, final_time, nt + 1)
    h, tau = x[1] - x[0], t[1] - t[0]
    # Центральные разности для первой и второй производных:
    # L_h U_j = minus*U_{j-1} + center*U_j + plus*U_{j+1}.
    minus = a / h**2 - b / (2 * h)
    center = c - 2 * a / h**2
    plus = a / h**2 + b / (2 * h)
    if theta == 0 and (min(minus, plus) < 0 or 1 + tau * center < -1e-14):
        raise ValueError(
            "Явная схема: нарушено условие неотрицательности весов. "
            "Уменьшите tau; при сильном переносе уменьшите также h."
        )

    # Для двухточечной границы второго порядка исключаем фиктивный узел
    # через центральную разность для u_x. Затем применяем исходное PDE
    # на границе. Получаем двухточечный оператор с учетом u_t, b*u_x, c*u.
    # Знак у 2*a/h разный, потому что слева фиктивный узел расположен
    # в x=-h, а справа — в x=L+h.
    def edge_operator(bc, side, time):
        sign = -1 if side == 0 else 1
        factor = sign * 2 * a / h + b
        d = -2 * a / h**2 + c - factor * bc.alpha / bc.beta
        neighbor = 2 * a / h**2
        forcing = factor * bc.value(time) / bc.beta + source(x[side], time)
        return d, neighbor, forcing

    if theta == 0 and boundary_order == "2p2":
        for side, bc in ((0, left_boundary), (-1, right_boundary)):
            if bc.beta != 0 and 1 + tau * edge_operator(bc, side, 0)[0] < -1e-14:
                raise ValueError("Явная схема: уменьшите tau для устойчивого шага на границе.")

    values = np.empty((nt + 1, nx + 1))
    values[0] = initial(x)
    for k in range(nt):
        old = values[k]
        now, next_time = t[k], t[k + 1]
        lower = np.zeros(nx + 1)
        diagonal = np.ones(nx + 1)
        upper = np.zeros(nx + 1)
        rhs = np.zeros(nx + 1)
        lower[1:-1] = -theta * tau * minus
        diagonal[1:-1] = 1 - theta * tau * center
        upper[1:-1] = -theta * tau * plus
        # (I - theta*tau*L_h) U^{k+1} =
        # (I + (1-theta)*tau*L_h) U^k + tau*f^{k+theta}.
        # Источник тоже взвешиваем по времени, иначе Кранк–Николсон
        # потеряет второй порядок для нестационарной правой части.
        rhs[1:-1] = old[1:-1] + tau * (
            (1 - theta) * (minus * old[:-2] + center * old[1:-1] + plus * old[2:])
            + (1 - theta) * source(x[1:-1], now)
            + theta * source(x[1:-1], next_time)
        )

        for side, bc in ((0, left_boundary), (-1, right_boundary)):
            if bc.beta == 0:
                # Условие Дирихле задается точно независимо от порядка границ.
                diagonal[side] = bc.alpha
                rhs[side] = bc.value(next_time)
            elif boundary_order == "2p2":
                d, neighbor, f_old = edge_operator(bc, side, now)
                f_new = edge_operator(bc, side, next_time)[2]
                j = 1 if side == 0 else -2
                diagonal[side] = 1 - theta * tau * d
                if side == 0:
                    upper[0] = -theta * tau * neighbor
                else:
                    lower[-1] = -theta * tau * neighbor
                rhs[side] = old[side] + tau * (
                    (1 - theta) * (d * old[side] + neighbor * old[j] + f_old)
                    + theta * f_new
                )
            else:
                # 2p1: u_x(0) ≈ (U1-U0)/h;
                # 3p2: u_x(0) ≈ (-3U0+4U1-U2)/(2h).
                # На правой границе используем зеркальные односторонние
                # формулы, но направление самой производной не меняем.
                sign = -1 if side == 0 else 1
                factor = 1 if boundary_order == "2p1" else 1.5
                diagonal[side] = bc.alpha + sign * factor * bc.beta / h
                adjacent = -sign * (1 if boundary_order == "2p1" else 2) * bc.beta / h
                extra = sign * bc.beta / (2 * h) if boundary_order == "3p2" else 0
                rhs[side] = bc.value(next_time)
                if theta == 0:
                    # В явной схеме внутренние узлы нового слоя уже известны
                    # (равны rhs). Поэтому границу находим простой подстановкой.
                    j, jj = (1, 2) if side == 0 else (-2, -3)
                    rhs[side] -= adjacent * rhs[j] + extra * rhs[jj]
                elif side == 0:
                    upper[0] = adjacent
                    if extra:
                        # Убираем U2 из граничной строки с помощью строки j=1.
                        # Матрица снова трехдиагональна; плотная СЛАУ не нужна.
                        if upper[1] == 0:
                            raise ArithmeticError("Нельзя исключить U2; измените пространственный шаг.")
                        multiplier = extra / upper[1]
                        diagonal[0] -= multiplier * lower[1]
                        upper[0] -= multiplier * diagonal[1]
                        rhs[0] -= multiplier * rhs[1]
                else:
                    lower[-1] = adjacent
                    if extra:
                        if lower[-2] == 0:
                            raise ArithmeticError("Нельзя исключить U[N-2]; измените пространственный шаг.")
                        multiplier = extra / lower[-2]
                        lower[-1] -= multiplier * diagonal[-2]
                        diagonal[-1] -= multiplier * upper[-2]
                        rhs[-1] -= multiplier * rhs[-2]

        # При theta=0 связи между неизвестными отсутствуют, поэтому прогонка
        # не требуется. Остальные две схемы решаем за O(nx) на временной слой.
        if theta == 0:
            if np.any(np.abs(diagonal) < 1e-14):
                raise ArithmeticError("Граничная формула вырождена на выбранной сетке.")
            values[k + 1] = rhs / diagonal
        else:
            values[k + 1] = tridiagonal_solve(lower, diagonal, upper, rhs)
    return x, t, values
