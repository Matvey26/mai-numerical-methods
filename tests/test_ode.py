import numpy as np

from nm.ode import (
    adams_bashforth_method,
    BoundaryCondition,
    euler_method,
    exact_solution_error,
    finite_difference_method,
    reduce_second_order_ode,
    runge_kutta_method,
    runge_romberg_grid_errors,
    runge_romberg_errors,
    shooting_method,
)


def variant_11_right_side(x, y, derivative):
    return y * np.cos(x) ** 2 - derivative * np.tan(x)


def variant_11_exact_solution(x):
    return np.exp(np.sin(x)) + np.exp(-np.sin(x))


def lab4_2_right_side(x, y, derivative):
    if np.isclose(x, 1.0):
        return 1.0
    return (x * derivative - y) / (x * (x - 1))


def lab4_2_exact_solution(x):
    return 1 + x + x * np.log(x)


def lab4_2_a(x):
    return x * (x - 1)


def lab4_2_b(x):
    return -x


def lab4_2_c(x):
    return 1.0


def lab4_2_f(x):
    return 0.0


def test_lab4_euler_method_variant_11():
    system = reduce_second_order_ode(variant_11_right_side)
    x, y = euler_method(system, 0.0, 1.0, 0.1, [2.0, 0.0])
    _, y_fine = euler_method(system, 0.0, 1.0, 0.05, [2.0, 0.0])

    assert np.allclose(x, np.linspace(0.0, 1.0, 11))
    assert np.allclose(
        y[:, 0],
        np.array([
            2.0000000000,
            2.0000000000,
            2.0200000000,
            2.0595999964,
            2.1177999772,
            2.1929969278,
            2.2829810294,
            2.3849386683,
            2.4954721944,
            2.6106471094,
            2.7260761990,
        ]),
    )
    assert np.isclose(
        runge_romberg_errors(y[:, 0], y_fine[:, 0], 0.1, 0.05, order=1)[-1],
        0.0140753824,
    )
    assert np.isclose(
        exact_solution_error(variant_11_exact_solution, x, y)[-1],
        0.0247765764,
    )


def test_lab4_runge_kutta_method_variant_11():
    system = reduce_second_order_ode(variant_11_right_side)
    x, y = runge_kutta_method(system, 0.0, 1.0, 0.1, [2.0, 0.0])
    _, y_fine = runge_kutta_method(system, 0.0, 1.0, 0.05, [2.0, 0.0])

    assert np.allclose(
        y[:, 0],
        np.array([
            2.0000000000,
            2.0099750209,
            2.0395995670,
            2.0879697475,
            2.1535729303,
            2.2342854823,
            2.3273824867,
            2.4295699518,
            2.5370498002,
            2.6456256500,
            2.7508530900,
        ]),
    )
    assert np.isclose(
        runge_romberg_errors(y[:, 0], y_fine[:, 0], 0.1, 0.05, order=4)[-1],
        0.0000000204,
    )
    assert np.isclose(
        exact_solution_error(variant_11_exact_solution, x, y)[-1],
        0.0000003147,
    )


def test_lab4_adams_bashforth_method_variant_11():
    system = reduce_second_order_ode(variant_11_right_side)
    x, y = adams_bashforth_method(system, 0.0, 1.0, 0.1, [2.0, 0.0])
    _, y_fine = adams_bashforth_method(system, 0.0, 1.0, 0.05, [2.0, 0.0])

    assert np.allclose(
        y[:, 0],
        np.array([
            2.0000000000,
            2.0099750209,
            2.0395995670,
            2.0879697475,
            2.1535744810,
            2.2342845248,
            2.3273664516,
            2.4295210319,
            2.5369437541,
            2.6454353961,
            2.7505525844,
        ]),
    )
    assert np.isclose(
        runge_romberg_errors(y[:, 0], y_fine[:, 0], 0.1, 0.05, order=4)[-1],
        0.0000183631,
    )
    assert np.isclose(
        exact_solution_error(variant_11_exact_solution, x, y)[-1],
        0.0003001909,
    )


def test_lab4_2_shooting_method_variant_11():
    left_boundary = BoundaryCondition(alpha=0.0, beta=1.0, value=2.0)
    right_boundary = BoundaryCondition(alpha=-1.0, beta=2.0, value=1.0)

    x, values, eta, phi = shooting_method(
        lab4_2_right_side,
        left=1.0,
        right=2.0,
        step=0.1,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        eta0=4.0,
        eta1=5.0,
        shoot_from='right',
    )
    _, fine_values, _, _ = shooting_method(
        lab4_2_right_side,
        left=1.0,
        right=2.0,
        step=0.05,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        eta0=4.0,
        eta1=5.0,
        shoot_from='right',
    )

    rr_errors = runge_romberg_grid_errors(values[:, 0], fine_values[:, 0], 0.1, 0.05, order=4)
    exact_errors = np.abs(values[:, 0] - lab4_2_exact_solution(x))

    assert np.isclose(phi(eta), 0.0)
    assert np.max(rr_errors) < 2e-4
    assert np.max(exact_errors) < 3e-3


def test_lab4_2_finite_difference_method_variant_11():
    left_boundary = BoundaryCondition(alpha=1.0, beta=0.0, value=2.0)
    right_boundary = BoundaryCondition(alpha=-1.0, beta=2.0, value=1.0)

    x, values = finite_difference_method(
        lab4_2_b,
        lab4_2_c,
        lab4_2_f,
        left=1.0,
        right=2.0,
        step=0.1,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        a=lab4_2_a,
    )
    _, fine_values = finite_difference_method(
        lab4_2_b,
        lab4_2_c,
        lab4_2_f,
        left=1.0,
        right=2.0,
        step=0.05,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        a=lab4_2_a,
    )

    rr_errors = runge_romberg_grid_errors(values, fine_values, 0.1, 0.05, order=2)
    exact_errors = np.abs(values - lab4_2_exact_solution(x))

    assert np.max(rr_errors) < 4e-4
    assert np.max(exact_errors) < 2e-3


def test_finite_difference_method_default_a_matches_explicit_one():
    def b(x):
        return 0.0

    def c(x):
        return -1.0

    def f(x):
        return 0.0

    left_boundary = BoundaryCondition(alpha=1.0, beta=0.0, value=0.0)
    right_boundary = BoundaryCondition(alpha=1.0, beta=0.0, value=np.sinh(1.0))

    _, default_a_values = finite_difference_method(
        b,
        c,
        f,
        left=0.0,
        right=1.0,
        step=0.1,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
    )
    _, explicit_a_values = finite_difference_method(
        b,
        c,
        f,
        left=0.0,
        right=1.0,
        step=0.1,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        a=1.0,
    )

    assert np.allclose(default_a_values, explicit_a_values)
