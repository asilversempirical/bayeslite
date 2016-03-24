from math import fsum
import numpy as np
from functools import partial

from hypothesis import given, assume
from hypothesis.strategies import floats

from util import arithmetic
reload(arithmetic)

def raises(f, args, kwargs, exception, msg):
    try:
        f(*args, **kwargs)
    except exception, e:
        assert e.args == (msg,)
    else:
        assert False, 'Failed to detect %s(%s) from %s(*%s, **%s)' % (
            exception, msg, f, args, kwargs)

@given(x=floats(), y=floats())
def test_safe_divide(x, y):
    r = partial(raises, arithmetic.safe_divide, (x, y), {})
    if abs(x) == abs(y) == np.inf:
        r(ValueError, "Can't divide infinity by infinity.")
    elif np.isnan(x) or np.isnan(y):
        r(ValueError, "Don't know how to deal with NaN")
    elif y != 0:
        assert np.isclose(arithmetic.safe_divide(x, y), x / y)
    else:
        r(ZeroDivisionError, "Can't divide by 0")

@given(x=floats())
def test_catastrophic_cancellation(x):
    assume(0 < abs(x) < 1e-8)
    answer = arithmetic.safe_divide(
        arithmetic.safe_divide(fsum([1, -np.cos(x)]), x), x)
    assert np.isclose(1, answer)

test_catastrophic_cancellation()
