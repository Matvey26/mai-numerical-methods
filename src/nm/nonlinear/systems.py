import numpy as np
from typing import Callable


def newton_system(
    funcs: list[Callable],
    x0: list[float] | np.ndarray,
    precision_epsilon: float | None = None,
    max_iter: int | None = 1000,
    h: float = 1e-6,
    jacobian: Callable | None = None,
) -> np.ndarray:
    """
    Метод Ньютона для системы нелинейных уравнений.

    Итерационный процесс
        x^(k+1) = x^(k) - J^(-1)(x^(k)) * f(x^(k))
    Критерий останова (формула 2.18):
        ||x^(k+1) - x^(k)|| <= epsilon

    Режимы остановки:
      - precision_epsilon задан, max_iter=None  -> останов по ||dx|| < epsilon
      - max_iter задан, precision_epsilon=None  -> ровно max_iter итераций
      - оба заданы                              -> останов по первому выполненному условию
      - ни одно не задано                       -> ValueError

    :param funcs:             Список функций [f1, f2, ..., fn], каждая принимает вектор x (np.ndarray).
    :param x0:                Начальное приближение (список или np.ndarray длины n).
    :param precision_epsilon: Требуемая точность по норме ||dx||.
    :param max_iter:          Максимальное число итераций.
    :param h:                 Шаг для численного дифференцирования якобиана (если jacobian=None).
    :param jacobian:          Аналитический якобиан J(x) -> np.ndarray shape (n, n).
                              Если None — якобиан вычисляется численно методом конечных разностей.
    :return:                  Вектор-решение np.ndarray shape (n,).
    :raises ValueError:       Если не задан ни один критерий останова, или размерности несогласованы.
    :raises np.linalg.LinAlgError: Если матрица Якоби вырождена (det ≈ 0).
    :raises RuntimeError:     Если метод не сошелся за отведенное число итераций.
    """
    # Валидация критериев останова
    if precision_epsilon is None and max_iter is None:
        raise ValueError(
            "[ERROR] Необходимо задать хотя бы один из параметров: "
            "precision_epsilon или max_iter."
        )

    # Подготовка начального приближения
    x_curr = np.array(x0, dtype=float)
    n = len(x_curr)

    if len(funcs) != n:
        raise ValueError(
            f"[ERROR] Число уравнений ({len(funcs)}) не совпадает с размерностью "
            f"начального приближения ({n}). Система должна быть квадратной."
        )

    if n == 0:
        raise ValueError("[ERROR] Система не может быть пустой.")

    # Вспомогательные функции
    def eval_f(x: np.ndarray) -> np.ndarray:
        """Вычисляет вектор f(x) = [f1(x), f2(x), ..., fn(x)]."""
        return np.array([f(x) for f in funcs], dtype=float)

    def _eval_jacobian_numerical(x: np.ndarray) -> np.ndarray:
        """
        Численный якобиан методом центральных разностей:
            J[i,j] = (fi(x + h*ej) - fi(x - h*ej)) / (2h)
        """
        J = np.zeros((n, n), dtype=float)
        for j in range(n):
            x_fwd = x.copy()
            x_bwd = x.copy()
            x_fwd[j] += h
            x_bwd[j] -= h
            J[:, j] = (eval_f(x_fwd) - eval_f(x_bwd)) / (2 * h)
        return J

    def get_jacobian(x: np.ndarray) -> np.ndarray:
        if jacobian is not None:
            J = np.array(jacobian(x), dtype=float)
            if J.shape != (n, n):
                raise ValueError(
                    f"[ERROR] Якобиан вернул матрицу формы {J.shape}, "
                    f"ожидается ({n}, {n})."
                )
            return J
        return _eval_jacobian_numerical(x)

    # Проверка начального приближения
    f0 = eval_f(x_curr)
    if not np.all(np.isfinite(f0)):
        raise ValueError(
            "[ERROR] Вектор f(x0) содержит nan или inf. "
            "Проверьте корректность функций и начального приближения."
        )

    J0 = get_jacobian(x_curr)
    det0 = np.linalg.det(J0)
    if abs(det0) < 1e-14:
        raise np.linalg.LinAlgError(
            f"[ERROR] Матрица Якоби в начальной точке вырождена "
            f"(det = {det0:.3e}). Метод Ньютона не применим."
        )

    print(
        f"[INFO] Начало итераций. n = {n}, x0 = {x_curr}, "
        f"||f(x0)|| = {np.linalg.norm(f0):.6e}, det(J(x0)) = {det0:.6e}."
    )
    if jacobian is None:
        print(f"[INFO] Якобиан вычисляется численно (h = {h}).")
    else:
        print("[INFO] Используется аналитический якобиан.")

    # Основной итерационный цикл
    iter_limit = max_iter if max_iter is not None else 10_000

    for iteration in range(iter_limit):
        f_val = eval_f(x_curr)
        J_val = get_jacobian(x_curr)

        # Проверка невырожденности якобиана на текущей итерации
        det_J = np.linalg.det(J_val)
        if abs(det_J) < 1e-14:
            raise np.linalg.LinAlgError(
                f"[ERROR] Матрица Якоби вырождена на итерации {iteration} "
                f"(det = {det_J:.3e}). Метод остановлен."
            )

        # Решаем СЛАУ: J(x^k) * dx^k = -f(x^k)  (формула 2.16)
        try:
            delta_x = np.linalg.solve(J_val, -f_val)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"[ERROR] Не удалось решить СЛАУ на итерации {iteration}: {e}"
            )

        # Обновление приближения: x^(k+1) = x^(k) + dx^(k)  (формула 2.15)
        x_next = x_curr + delta_x

        # Проверка на nan/inf после шага
        if not np.all(np.isfinite(x_next)):
            raise RuntimeError(
                f"[ERROR] Итерация {iteration}: x_next содержит nan или inf. "
                "Метод расходится. Проверьте начальное приближение."
            )

        # Критерий останова по точности (формула 2.18): ||x^(k+1) - x^(k)|| <= eps
        norm_delta = np.linalg.norm(delta_x)
        if precision_epsilon is not None and norm_delta <= precision_epsilon:
            print(
                f"[INFO] Сходимость достигнута на итерации {iteration + 1}. "
                f"||dx|| = {norm_delta:.3e} <= eps = {precision_epsilon:.3e}. "
                f"||f(x*)|| = {np.linalg.norm(eval_f(x_next)):.3e}."
            )
            return x_next

        x_curr = x_next

    # Завершение по max_ite
    if precision_epsilon is None:
        f_final = eval_f(x_curr)
        print(
            f"[INFO] Выполнено {iter_limit} итераций (режим фиксированного числа итераций). "
            f"||f(x*)|| = {np.linalg.norm(f_final):.3e}."
        )
        return x_curr

    raise RuntimeError(
        f"[ERROR] Метод Ньютона не сошелся за {iter_limit} итераций. "
        f"||f(x_last)|| = {np.linalg.norm(eval_f(x_curr)):.3e}. "
        "Проверьте начальное приближение или увеличьте max_iter."
    )


def simple_iteration_system(
    phis: list[Callable],
    x0: list[float] | np.ndarray,
    precision_epsilon: float | None = None,
    max_iter: int | None = 1000,
    q: float | str | None = "auto",
    h: float = 1e-6,
) -> np.ndarray:
    """
    Метод простой итерации для системы нелинейных уравнений.
 
    Режимы остановки:
      - precision_epsilon задан, max_iter=None  -> останов по апостериорной оценке (с учётом q)
      - max_iter задан, precision_epsilon=None  -> ровно max_iter итераций, q игнорируется
      - оба заданы                              -> останов по первому выполненному условию
      - ни одно не задано                       -> ValueError
 
    :param phis:              Список итерирующих функций [phi1, phi2, ..., phin],
                              каждая принимает вектор x (np.ndarray) и возвращает float.
    :param x0:                Начальное приближение (список или np.ndarray длины n).
    :param precision_epsilon: Требуемая точность по норме ||x^(k+1) - x^(k)||.
    :param max_iter:          Максимальное число итераций.
    :param q:                 Константа сжатия (норма матрицы Якоби phi'(x)).
                              'auto' — вычислить численно в окрестности x0 (по умолчанию).
                              float  — использовать переданное значение без проверок.
                              None   — использовать наивный критерий ||dx|| < epsilon.
    :param h:                 Шаг для численного дифференцирования якобиана phi'(x).
    :return:                  Вектор-решение np.ndarray shape (n,).
    :raises ValueError:       Если не задан ни один критерий останова, или размерности несогласованы,
                              или q >= 1 при явном задании.
    :raises RuntimeError:     Если метод не сошелся за отведенное число итераций.
    """
    # Валидация критериев останова
    if precision_epsilon is None and max_iter is None:
        raise ValueError(
            "[ERROR] Необходимо задать хотя бы один из параметров: "
            "precision_epsilon или max_iter."
        )
 
    # Подготовка начального приближения
    x_curr = np.array(x0, dtype=float)
    n = len(x_curr)
 
    if len(phis) != n:
        raise ValueError(
            f"[ERROR] Число итерирующих функций phi ({len(phis)}) не совпадает с размерностью "
            f"начального приближения ({n}). Система должна быть квадратной."
        )
 
    if n == 0:
        raise ValueError("[ERROR] Система не может быть пустой.")
 
    # Вспомогательные функции
    def eval_phi(x: np.ndarray) -> np.ndarray:
        """Вычисляет вектор phi(x) = [phi1(x), phi2(x), ..., phin(x)]."""
        return np.array([phi(x) for phi in phis], dtype=float)
 
    def eval_phi_jacobian(x: np.ndarray) -> np.ndarray:
        """
        Численный якобиан phi'(x) методом центральных разностей:
            J[i,j] = (phii(x + h*ej) - phii(x - h*ej)) / (2h)
        """
        J = np.zeros((n, n), dtype=float)
        for j in range(n):
            x_fwd = x.copy()
            x_bwd = x.copy()
            x_fwd[j] += h
            x_bwd[j] -= h
            J[:, j] = (eval_phi(x_fwd) - eval_phi(x_bwd)) / (2 * h)
        return J
 
    # Проверка начального приближения
    phi0 = eval_phi(x_curr)
    if not np.all(np.isfinite(phi0)):
        raise ValueError(
            "[ERROR] Вектор phi(x0) содержит nan или inf. "
            "Проверьте корректность итерирующих функций и начального приближения."
        )
 
    # Проверка условия сходимости и вычисление q
    # (только если нужна проверка по точности)
    stop_criterion = None
 
    if precision_epsilon is not None:
        q_val = None
 
        if q == "auto":
            J_phi = eval_phi_jacobian(x_curr)
            q_val = np.linalg.norm(J_phi, ord=2)
            print(
                f"[INFO] Автоматически вычисленное q = ||phi'(x0)|| = {q_val:.6f} "
                f"(численное дифференцирование, h = {h})."
            )
        elif isinstance(q, (float, int)):
            q_val = float(q)
            if q_val < 0:
                raise ValueError(
                    f"[ERROR] Константа сжатия q должна быть неотрицательной, получено q = {q_val}."
                )
            print(f"[INFO] Используется заданное пользователем q = {q_val:.6f}.")
        # q is None -> наивный критерий
 
        if q_val is not None:
            if q_val >= 1.0:
                print(
                    f"[WARNING] Условие сходимости (Теорема 2.4) НЕ выполнено: "
                    f"q = {q_val:.6f} >= 1. Апостериорная оценка (2.26) не применима. "
                    "Используется наивный критерий останова ||dx|| < epsilon. "
                    "Метод может расходиться!"
                )
                stop_criterion = precision_epsilon
            else:
                # Апостериорная оценка: ||x* - x^(k+1)|| <= q/(1-q) * ||x^(k+1) - x^(k)||
                # Останавливаемся, когда: ||dx|| < epsilon * (1 - q) / q
                stop_criterion = precision_epsilon * (1.0 - q_val) / q_val
                print(
                    f"[INFO] Условие сходимости выполнено: q = {q_val:.6f} < 1. "
                    f"Скорректированный порог останова по (2.26): "
                    f"||dx|| < {stop_criterion:.3e}."
                )
        else:
            # q не задан — наивный критерий
            stop_criterion = precision_epsilon
            print(
                "[INFO] q не задан. Используется наивный критерий останова: "
                f"||dx|| < epsilon = {precision_epsilon:.3e}."
            )
 
    print(
        f"[INFO] Начало итераций. n = {n}, x0 = {x_curr}, "
        f"||phi(x0) - x0|| = {np.linalg.norm(phi0 - x_curr):.6e}."
    )
 
    # Основной итерационный цикл
    iter_limit = max_iter if max_iter is not None else 10_000
 
    for iteration in range(iter_limit):
        x_next = eval_phi(x_curr)
 
        # Проверка на nan/inf после шага
        if not np.all(np.isfinite(x_next)):
            raise RuntimeError(
                f"[ERROR] Итерация {iteration}: x_next содержит nan или inf. "
                "Метод расходится. Проверьте итерирующие функции phi и начальное приближение."
            )
 
        # Критерий останова по точности с учётом апостериорной оценки (2.26)
        norm_delta = np.linalg.norm(x_next - x_curr)
        if stop_criterion is not None and norm_delta < stop_criterion:
            print(
                f"[INFO] Сходимость достигнута на итерации {iteration + 1}. "
                f"||dx|| = {norm_delta:.3e} < порог = {stop_criterion:.3e}."
            )
            return x_next
 
        x_curr = x_next
 
    # Завершение по max_iter
    if precision_epsilon is None:
        print(
            f"[INFO] Выполнено {iter_limit} итераций (режим фиксированного числа итераций). "
            f"||phi(x*) - x*|| = {np.linalg.norm(eval_phi(x_curr) - x_curr):.3e}."
        )
        return x_curr
 
    raise RuntimeError(
        f"[ERROR] Метод простой итерации не сошелся за {iter_limit} итераций. "
        f"||dx_last|| = {np.linalg.norm(eval_phi(x_curr) - x_curr):.3e}. "
        "Проверьте итерирующие функции phi или увеличьте max_iter."
    )
 