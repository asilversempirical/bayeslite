import numpy as np
from numbers import Real, Integral
import sys

# The smallest possible positive float
minfloat = sys.float_info.min*sys.float_info.epsilon


def isintegral(v):
    return isinstance(v, Integral)


def isreal(v):
    return isinstance(v, Real)


def ispositive(v):
    return isreal(v) and v > 0


def isfinite(v):
    return isreal(v) and (-np.inf < v < np.inf)
