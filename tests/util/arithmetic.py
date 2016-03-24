"""Safe arithmetic operations, avoiding catastrophic cancellation or silent
over/underflow.

XXX: Note that this doesn't actually work to avoid catastrophic cancellation.
See test.test_arithmetic.test_catastrophic_cancellation.

"""

from math import fsum
from numpy import log, exp, abs, prod, sign, inf, isnan


def safe_product(factors):
    """Return product, avoiding catastrophic cancellation or silent
    over/underflow.

    """
    sgn = 1  # Accumulates the sign (+/-1) of factors so far
    logsum = 0  # Accumulates the log of the product of the factors so far
    # Iterating over factors explicitly like this allows for iterators which
    # don't expect to have to provide their values more than once.
    for factor in factors:
        if factor == 0:  # Can't take log of 0
            return 0
        sgn *= sign(factor)
        logsum = fsum(logsum, log(abs(factor)))
    return sgn * exp(logsum)


def safe_divide(dividend, divisor):
    """Return quotient, avoiding catastrophic cancellation or silent
    over/underflow."""
    if isnan(dividend) or isnan(divisor):
        raise ValueError("Don't know how to deal with NaN")
    if divisor == 0:
        raise ZeroDivisionError("Can't divide by 0")
    if abs(dividend) == abs(divisor) == inf:
        raise ValueError("Can't divide infinity by infinity.")
    if dividend == 0:  # Can't take log of 0
        return 0
    return (sign(dividend) * sign(divisor) *
            exp(fsum([log(abs(dividend)), -log(abs(divisor))])))


def safe_mean(data):
    """Return mean, avoiding catastrophic cancellation or silent
    over/underflow.

    'data' must allow for multiple iterations over its values, and have
    a length.

    """
    if len(data) == 0:
        raise ValueError("Can't take mean of empty list")
    return safe_divide(fsum(data), len(data))
