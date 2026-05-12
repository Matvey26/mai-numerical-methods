import numpy as np

from nm.ode import (
    adams_bashforth_method,
    euler_method,
    exact_solution_error,
    reduce_second_order_ode,
    runge_kutta_method,
    runge_romberg_errors,
)


def variant_11_right_side(x, y, derivative):
    return y * np.cos(x) ** 2 - derivative * np.tan(x)


def variant_11_exact_solution(x):
    return np.exp(np.sin(x)) + np.exp(-np.sin(x))


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
