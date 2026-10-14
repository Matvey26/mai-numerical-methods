"""Проверка варианта 8 и порядков аппроксимации граничных условий.

Запуск: python -m unittest discover -s tests -v
"""
import unittest

import numpy as np

from nm.pde.parabolic import BoundaryCondition, solve_parabolic


class ParabolicTests(unittest.TestCase):
    def test_variant8_all_methods_and_boundaries(self):
        left = BoundaryCondition(0, 1, lambda t: np.exp(-2*t))
        right = BoundaryCondition(1, 0, lambda t: np.exp(-2*t))
        for theta in (0, 1, 0.5):
            for mode in ('2p1', '3p2', '2p2'):
                with self.subTest(theta=theta, mode=mode):
                    errors = []
                    for n in (20, 40):
                        x, t, u = solve_parabolic(
                            1, 0, -1, lambda x, t: np.zeros_like(x), np.sin,
                            left, right, 0, np.pi/2, 0.1, n, 2*n*n, theta, mode,
                        )
                        np.testing.assert_allclose(u[:, -1], np.exp(-2*t), atol=1e-12)
                        exact = np.exp(-2*t[:, None])*np.sin(x)
                        errors.append(np.max(np.abs(u - exact)))
                    self.assertLess(errors[1], 0.4*errors[0])
                    self.assertLess(errors[1], 1e-3)

    def test_boundary_orders_without_sine_cancellation(self):
        length = np.pi/2
        initial = lambda x: 1+x+x*x+np.sin(x)
        source = lambda x, t: np.exp(t)*(2*initial(x) - (2-np.sin(x)))
        left = BoundaryCondition(0, 1, lambda t: 2*np.exp(t))
        right = BoundaryCondition(1, 0, lambda t: initial(length)*np.exp(t))
        for theta in (0, 1, 0.5):
            for mode in ('2p1', '3p2', '2p2'):
                with self.subTest(theta=theta, mode=mode):
                    errors = []
                    for n in (40, 80):
                        x, t, u = solve_parabolic(
                            1, 0, -1, source, initial, left, right,
                            0, length, 0.1, n, 2*n*n, theta, mode,
                        )
                        exact = np.exp(t[:, None])*initial(x)
                        errors.append(np.max(np.abs(u - exact)))
                    order = np.log2(errors[0]/errors[1])
                    expected = 1 if mode == '2p1' else 2
                    self.assertAlmostEqual(order, expected, delta=0.15)

    def test_explicit_reaction_term_in_step_restriction(self):
        # sigma=1/2 satisfies the diffusion-only bound, but c<0 makes
        # the central explicit weight negative; the solver must reject it.
        with self.assertRaises(ValueError):
            solve_parabolic(
                1, 0, -1, lambda x, t: np.zeros_like(x), np.sin,
                BoundaryCondition(0, 1, lambda t: np.exp(-2*t)),
                BoundaryCondition(1, 0, lambda t: np.exp(-2*t)),
                0, 1, 0.005, 10, 1, 0, '2p2',
            )


if __name__ == '__main__':
    unittest.main()
