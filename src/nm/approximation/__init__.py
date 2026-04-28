from .interpolation import (
    divided_differences,
    lagrange_polynomial,
    newton_polynomial,
)
from .splines import (
    evaluate_cubic_spline,
    natural_cubic_spline_coefficients,
)

__all__ = [
    'divided_differences',
    'lagrange_polynomial',
    'newton_polynomial',
    'evaluate_cubic_spline',
    'natural_cubic_spline_coefficients',
]
