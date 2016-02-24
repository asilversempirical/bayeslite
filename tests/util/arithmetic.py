"""Safe arithmetic operations, avoiding catastrophic cancellation or silent
over/underflow."""

from math import fsum
from numpy import log, exp, abs, prod, sign


def safe_product(factors):
    """Return product, avoiding catastrophic cancellation or silent
    over/underflow.

    "factors" must allow for multiple iterations over its values.

    """
    if 0 in factors:  # Can't take log of 0.
        return 0
    return prod(sign(factors)) * exp(fsum(log(abs(factors))))


def safe_divide(dividend, divisor):
    """Return quotient, avoiding catastrophic cancellation or silent
    over/underflow."""
    if dividend == 0:
        return 0
    assert dividend != 0, "Can't take log of 0"
    return sign(dividend) * exp(fsum([log(abs(dividend)), -log(divisor)]))


def safe_mean(data):
    """Return mean, avoiding catastrophic cancellation or silent
    over/underflow.

    'data' must allow for multiple iterations over its values, and have
    a length.

    """
    return safe_divide(fsum(data), len(data))
