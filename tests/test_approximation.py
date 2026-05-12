import numpy as np

from nm.approximation import (
    interpolation_actual_error,
    interpolation_aposteriori_error_estimate,
    interpolation_apriori_error_bound,
    evaluate_cubic_spline,
    lagrange_polynomial,
    least_squares_normal_system,
    least_squares_polynomial,
    natural_cubic_spline_coefficients,
    newton_polynomial,
    rectangle_error_bound,
    rectangle_rule,
    runge_romberg_error,
    simpson_error_bound,
    simpson_rule,
    squared_error_sum,
    trapezoidal_error_bound,
    trapezoidal_rule,
)


def arcctg(x):
    return np.pi / 2 - np.arctan(x)


def arcctg_fourth_derivative(x):
    return 24 * x * (x ** 2 - 1) / (1 + x ** 2) ** 4


def test_lagrange_and_newton_polynomials_match_on_variant_11():
    x_nodes = np.array([-3.0, -1.0, 1.0, 3.0])
    y_nodes = arcctg(x_nodes)

    lagrange = lagrange_polynomial(x_nodes, y_nodes)
    newton = newton_polynomial(x_nodes, y_nodes)

    assert np.allclose(lagrange(x_nodes), y_nodes)
    assert np.allclose(newton(x_nodes), y_nodes)
    assert np.allclose(lagrange.coeffs, newton.coeffs)


def test_actual_interpolation_error_for_variant_11():
    x_star = -0.5

    for x_nodes in (
        np.array([-3.0, -1.0, 1.0, 3.0]),
        np.array([-3.0, 0.0, 1.0, 3.0]),
    ):
        y_nodes = arcctg(x_nodes)
        polynomial = lagrange_polynomial(x_nodes, y_nodes)

        assert np.isclose(polynomial(x_star), 1.9807946072)
        assert np.isclose(interpolation_actual_error(arcctg, polynomial, x_star), 0.0536493286)


def test_apriori_error_bound_uses_derivative_of_next_order():
    x_nodes = np.array([0.0, 1.0, 2.0, 3.0])
    y_nodes = x_nodes ** 4
    x_star = 1.5
    polynomial = lagrange_polynomial(x_nodes, y_nodes)

    actual_error = interpolation_actual_error(lambda x: x ** 4, polynomial, x_star)
    apriori_error = interpolation_apriori_error_bound(
        x_nodes,
        x_star,
        derivative=lambda x: np.full_like(x, 24.0),
    )

    assert np.isclose(actual_error, 0.5625)
    assert np.isclose(apriori_error, actual_error)


def test_aposteriori_error_is_first_omitted_newton_term():
    x_nodes = np.array([0.0, 1.0, 2.0, 3.0])
    y_nodes = x_nodes ** 4
    x_star = 1.5

    aposteriori_error = interpolation_aposteriori_error_estimate(
        x_nodes,
        y_nodes,
        x_star,
        extra_x=4.0,
        extra_y=4.0 ** 4,
    )

    assert np.isclose(aposteriori_error, 0.5625)


def test_variant_11_apriori_bound_covers_actual_error():
    x_nodes = np.array([-3.0, -1.0, 1.0, 3.0])
    y_nodes = arcctg(x_nodes)
    x_star = -0.5
    polynomial = lagrange_polynomial(x_nodes, y_nodes)

    actual_error = interpolation_actual_error(arcctg, polynomial, x_star)
    apriori_error = interpolation_apriori_error_bound(
        x_nodes,
        x_star,
        derivative=arcctg_fourth_derivative,
    )

    assert apriori_error >= actual_error


def test_lab3_2_natural_spline_uses_expected_coefficients():
    x_nodes = np.array([-3.0, -1.0, 1.0, 3.0, 5.0])
    y_nodes = np.array([2.8198, 2.3562, 0.78540, 0.32175, 0.19740])
    coefficients = natural_cubic_spline_coefficients(x_nodes, y_nodes)
    a, b, c, d = coefficients

    left_second_derivative = 2 * c[0]
    last_h = x_nodes[-1] - x_nodes[-2]
    right_second_derivative = 2 * c[-1] + 6 * d[-1] * last_h

    assert np.isclose(left_second_derivative, 0.0)
    assert np.isclose(right_second_derivative, 0.0)
    for x_node, y_node in zip(x_nodes, y_nodes):
        assert np.isclose(evaluate_cubic_spline(x_nodes, coefficients, x_node), y_node)
    assert np.isclose(evaluate_cubic_spline(x_nodes, coefficients, -0.5), 1.9977782645)


def test_lab3_3_least_squares_variant_11():
    x_nodes = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
    y_nodes = np.array([2.9442, 2.8198, 2.3562, 0.7854, 0.32175, 0.1974])

    linear = least_squares_polynomial(x_nodes, y_nodes, degree=1)
    quadratic = least_squares_polynomial(x_nodes, y_nodes, degree=2)
    cubic = least_squares_polynomial(x_nodes, y_nodes, degree=3)

    assert np.allclose(linear.coeffs, np.array([-0.3256992857, 1.5707916667]))
    assert np.allclose(
        quadratic.coeffs,
        np.array([0.0000002232, -0.3256992857, 1.5707890625]),
    )
    assert np.allclose(
        cubic.coeffs,
        np.array([0.0116152199, 0.0000002232, -0.5603267280, 1.5707890625]),
    )
    assert np.isclose(squared_error_sum(linear, x_nodes, y_nodes), 0.7006866120)
    assert np.isclose(squared_error_sum(quadratic, x_nodes, y_nodes), 0.7006866120)
    assert np.isclose(squared_error_sum(cubic, x_nodes, y_nodes), 0.1411740353)


def test_lab3_3_least_squares_normal_system_variant_11():
    x_nodes = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
    y_nodes = np.array([2.9442, 2.8198, 2.3562, 0.7854, 0.32175, 0.1974])

    normal_matrix, right_side = least_squares_normal_system(x_nodes, y_nodes, degree=2)

    assert np.allclose(
        normal_matrix,
        np.array([
            [6.0, 0.0, 70.0],
            [0.0, 70.0, 0.0],
            [70.0, 0.0, 1414.0],
        ]),
    )
    assert np.allclose(right_side, np.array([9.42475, -22.79895, 109.95555]))


def test_lab3_5_numerical_integration_variant_11():
    function = lambda x: 1 / (x ** 3 + 64)

    rectangle_h1 = rectangle_rule(function, -2.0, 2.0, 1.0)
    rectangle_h2 = rectangle_rule(function, -2.0, 2.0, 0.5)
    trapezoidal_h1 = trapezoidal_rule(function, -2.0, 2.0, 1.0)
    trapezoidal_h2 = trapezoidal_rule(function, -2.0, 2.0, 0.5)
    simpson_h1 = simpson_rule(function, -2.0, 2.0, 1.0)
    simpson_h2 = simpson_rule(function, -2.0, 2.0, 0.5)

    assert np.isclose(rectangle_h1, 0.0625872651)
    assert np.isclose(rectangle_h2, 0.0626255883)
    assert np.isclose(trapezoidal_h1, 0.0627556471)
    assert np.isclose(trapezoidal_h2, 0.0626714561)
    assert np.isclose(simpson_h1, 0.0626755189)
    assert np.isclose(simpson_h2, 0.0626433925)

    assert np.isclose(
        abs(runge_romberg_error(rectangle_h1, rectangle_h2, 1.0, 0.5, order=2)),
        0.0000127744,
    )
    assert np.isclose(
        abs(runge_romberg_error(trapezoidal_h1, trapezoidal_h2, 1.0, 0.5, order=2)),
        0.0000280637,
    )
    assert np.isclose(
        abs(runge_romberg_error(simpson_h1, simpson_h2, 1.0, 0.5, order=4)),
        0.0000021418,
    )


def test_lab3_5_theoretical_error_bounds_variant_11():
    second_derivative_bound = 0.0054664723032070
    fourth_derivative_bound = 0.0154288391741536

    assert np.isclose(
        rectangle_error_bound(-2.0, 2.0, 1.0, second_derivative_bound),
        0.0009110787,
    )
    assert np.isclose(
        trapezoidal_error_bound(-2.0, 2.0, 1.0, second_derivative_bound),
        0.0018221574,
    )
    assert np.isclose(
        simpson_error_bound(-2.0, 2.0, 1.0, fourth_derivative_bound),
        0.0003428631,
    )
