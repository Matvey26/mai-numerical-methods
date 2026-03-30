import numpy as np
from typing import Callable


def dichotomy(
    func: Callable,
    interval: tuple[float, float],
    precision_epsilon: float | None = None,
    max_iter: int | None = None,
) -> float:
    """
    Метод дихотомии (бисекции).

    Режимы остановки:
      - precision_epsilon задан, max_iter=None  → останов по точности (right-left)/2 < eps
      - max_iter задан, precision_epsilon=None  → ровно max_iter итераций, точность игнорируется
      - оба заданы                              → останов по первому выполненному условию
      - ни одно не задано                       → ValueError
    """
    if precision_epsilon is None and max_iter is None:
        raise ValueError(
            "[ERROR] Необходимо задать хотя бы один из параметров: "
            "precision_epsilon или max_iter."
        )

    left, right = interval

    if left > right:
        left, right = right, left
        print("[WARNING] Интервал был автоматически развернут.")

    left_value = func(left)
    right_value = func(right)

    if np.sign(left_value) == np.sign(right_value):
        raise ValueError(
            "[ERROR] Функция должна иметь разные знаки на концах интервала."
        )

    iteration = 0
    while True:
        # Проверка условия останова по итерациям
        if max_iter is not None and iteration >= max_iter:
            break

        # Проверка условия останова по точности
        if precision_epsilon is not None and (right - left) / 2 <= precision_epsilon:
            break

        mid = (left + right) / 2
        mid_value = func(mid)

        if mid_value == 0:
            return mid

        if np.sign(left_value) != np.sign(mid_value):
            right = mid
            right_value = mid_value
        elif np.sign(mid_value) != np.sign(right_value):
            left = mid
            left_value = mid_value
        else:
            raise ValueError(
                "[ERROR] Dichotomy method: Некорректно задан начальный интервал."
            )

        iteration += 1

    return (left + right) / 2


def newton(
    func: Callable,
    deriv: Callable,
    x0: float,
    precision_epsilon: float | None = None,
    max_iter: int | None = 1000,
    h: float = 1e-5,
) -> float:
    """
    Метод Ньютона (касательных).

    Режимы остановки:
      - precision_epsilon задан, max_iter=None  → останов по |x_next - x_curr| < eps
      - max_iter задан, precision_epsilon=None  → ровно max_iter итераций
      - оба заданы                              → останов по первому выполненному условию
      - ни одно не задано                       → ValueError
    """
    if precision_epsilon is None and max_iter is None:
        raise ValueError(
            "[ERROR] Необходимо задать хотя бы один из параметров: "
            "precision_epsilon или max_iter."
        )

    f_x0 = func(x0)
    f_double_prime = (func(x0 + h) - 2 * f_x0 + func(x0 - h)) / (h ** 2)
    condition_value = f_x0 * f_double_prime

    if condition_value <= 0:
        print(
            f"[WARNING] Условие f(x0)*f''(x0) > 0 не выполнено "
            f"(значение: {condition_value:.6f}). "
            f"Теорема 2.2 не гарантирует монотонную сходимость. Метод может разойтись!"
        )
    else:
        print("[INFO] Достаточное условие сходимости выполнено. Метод гарантированно сойдется.")

    x_curr = x0
    iter_limit = max_iter if max_iter is not None else 10_000

    for _ in range(iter_limit):
        f_val = func(x_curr)
        df_val = deriv(x_curr)

        if abs(df_val) <= 1e-6:
            raise ZeroDivisionError(
                "[ERROR] Производная равна нулю. Метод Ньютона не может быть применен."
            )

        x_next = x_curr - f_val / df_val

        if precision_epsilon is not None and abs(x_next - x_curr) < precision_epsilon:
            return x_next

        x_curr = x_next

    # Если дошли сюда — либо исчерпали max_iter (режим "только итерации"), либо не сошлись
    if precision_epsilon is None:
        return x_curr

    raise RuntimeError(
        "[ERROR] Метод Ньютона не сошелся за отведенное число итераций. "
        "Проверьте начальное приближение x0."
    )


def secant(
    func: Callable,
    x0: float,
    x1: float,
    precision_epsilon: float | None = None,
    max_iter: int | None = 1000,
) -> float:
    """
    Метод секущих.

    Режимы остановки:
      - precision_epsilon задан, max_iter=None  → останов по |x_next - x_curr| < eps
      - max_iter задан, precision_epsilon=None  → ровно max_iter итераций
      - оба заданы                              → останов по первому выполненному условию
      - ни одно не задано                       → ValueError
    """
    if precision_epsilon is None and max_iter is None:
        raise ValueError(
            "[ERROR] Необходимо задать хотя бы один из параметров: "
            "precision_epsilon или max_iter."
        )

    x_prev = x0
    x_curr = x1
    f_prev = func(x_prev)
    iter_limit = max_iter if max_iter is not None else 10_000

    for _ in range(iter_limit):
        f_curr = func(x_curr)

        if f_curr == f_prev:
            raise ZeroDivisionError(
                "[ERROR] Разность f(x_curr) - f(x_prev) равна нулю. "
                "Метод секущих не может продолжить вычисления."
            )

        x_next = x_curr - (f_curr * (x_curr - x_prev)) / (f_curr - f_prev)

        if precision_epsilon is not None and abs(x_next - x_curr) < precision_epsilon:
            return x_next

        x_prev, x_curr = x_curr, x_next
        f_prev = f_curr
        if x_prev == x_curr:
            break

    if precision_epsilon is None:
        return x_curr

    raise RuntimeError(
        "[ERROR] Метод секущих не сошелся за отведенное число итераций."
    )


def simple_iteration(
    func: Callable,
    x0: float,
    precision_epsilon: float | None = None,
    q: float | str | None = None,
    lmb: float | None = None,
    interval: tuple[float, float] | None = None,
    max_iter: int | None = 1000,
) -> float:
    """
    Метод простой итерации с автоматическим расчетом q и релаксацией через lambda.

    Режимы остановки:
      - precision_epsilon задан, max_iter=None  → останов по критерию (с учётом q)
      - max_iter задан, precision_epsilon=None  → ровно max_iter итераций, q игнорируется
      - оба заданы                              → останов по первому выполненному условию
      - ни одно не задано                       → ValueError

    :param func:            Исходная функция f(x) или итерирующая phi(x) (зависит от lmb).
    :param x0:              Начальное приближение.
    :param precision_epsilon: Требуемая точность вычислений.
    :param q:               Константа сжатия Липшица.
                            'auto'  — вычислить численно максимум производной.
                            float   — использовать переданное число без проверок.
                            None    — использовать наивный критерий остановки.
    :param lmb:             Параметр релаксации. Если задан, phi(x) = x - lmb * func(x).
    :param interval:        Отрезок [a, b] для поиска максимума производной при q='auto'.
    :param max_iter:        Лимит итераций.
    :return:                Найденный корень.
    """
    if precision_epsilon is None and max_iter is None:
        raise ValueError(
            "[ERROR] Необходимо задать хотя бы один из параметров: "
            "precision_epsilon или max_iter."
        )

    left, right = interval if interval is not None else (x0 - 1.0, x0 + 1.0)

    if lmb == "auto":
        x_grid = np.linspace(left, right, 100)
        h = 1e-5
        try:
            derivs = (func(x_grid + h) - func(x_grid - h)) / (2 * h)
        except Exception:
            derivs = np.array(
                [(func(x + h) - func(x - h)) / (2 * h) for x in x_grid]
            )
        if np.min(derivs) <= 0:
            derivs = np.abs(derivs)
        lmb = 2 / (np.max(derivs) + np.min(derivs))
        print(f"[INFO] Автоматически вычисленное lmb = {lmb:.6f}")

    # Формируем итерирующую функцию phi(x)
    phi = (lambda x: x - lmb * func(x)) if lmb is not None else func

    # Определяем stop_criterion только если нужна проверка по точности
    stop_criterion = None
    if precision_epsilon is not None:
        q_val = None

        if q == "auto":
            x_grid = np.linspace(left, right, 100)
            h = 1e-5
            try:
                derivs = (phi(x_grid + h) - phi(x_grid - h)) / (2 * h)
            except Exception:
                derivs = np.array(
                    [(phi(x + h) - phi(x - h)) / (2 * h) for x in x_grid]
                )
            q_val = np.max(np.abs(derivs))
            print(f"[INFO] Автоматически вычисленное q (max|phi'(x)|) = {q_val:.6f}")

        elif isinstance(q, (float, int)):
            q_val = float(q)

        if q_val is not None:
            if q_val >= 1.0:
                print(
                    "[WARNING] q >= 1. Формула апостериорной оценки не применима. "
                    "Используется наивный критерий."
                )
                stop_criterion = precision_epsilon
            else:
                stop_criterion = precision_epsilon * (1 - q_val) / q_val
        else:
            stop_criterion = precision_epsilon

    iter_limit = max_iter if max_iter is not None else 10_000
    x_curr = x0

    for _ in range(iter_limit):
        x_next = phi(x_curr)

        if stop_criterion is not None and abs(x_next - x_curr) < stop_criterion:
            return x_next

        x_curr = x_next

    if precision_epsilon is None:
        return x_curr

    raise RuntimeError(
        "[ERROR] Метод простой итерации не сошелся за отведенное число итераций."
    )
