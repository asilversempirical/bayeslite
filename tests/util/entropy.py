"""Wrap random variates to enforce explicit settings of PRNG seeds.

To use a distribution from scipy.stats, refer to it via rvs, e.g.,
n=rvs.norm(0,1) will return a normal distribution.

"""

from functools import partial
from hypothesis import strategies as st
import numbers
from numpy.random import RandomState
from numpy import ndarray
from scipy import stats as original_stats

word_mask = 2**32 - 1  # RandomState can only accept 32-bit seeds

class HighEntropyRandomState(RandomState):

    "A RandomState which has been seeded with at least 64 bits of entropy"


def random_state(seed):
    """While RandomState can only accept 32-bit seeds, it can take an array of
    them. This function produces such an array and passes it to RandomState.
    This is important because if we only have 32 bits of entropy to draw on we
    will run into birthday paradoxes in massively parallel situations.

    """
    if (seed < 0) or (not isinstance(seed, numbers.Integral)):
        raise ValueError('Seed must be a non-negative integer')
    seed_array = []
    while seed:
        seed_array.append(seed & word_mask)
        seed >>= 32
    # Make sure we're using at least a 64-bit seed (at least 2 32-bit numbers)
    seed_array.extend(max(0, 2 - len(seed_array)) * [0])
    return HighEntropyRandomState(seed_array)

seeds = partial(st.integers, min_value=0)

class StatsWrapper:

    def __init__(self, *args, **kwargs):
        vname = kwargs.pop('vname')
        cls = getattr(original_stats, vname)
        self.__obj = cls(*args, **kwargs)

    def __getattr__(self, name):
        if name != 'rvs':
            return getattr(self.__obj, name)
        return self.rvs

    def rvs(self, size=None, random_state=None):
        if not isinstance(random_state, HighEntropyRandomState):
            raise ValueError('Need to explicitly set random_state with '
                             'util.entropy.HighEntropyRandomState instance')
        # standard rvs method mutates a copy of the kwds dictionary...
        rv = self.__obj.rvs(size=size, random_state=random_state)
        if type(rv) == ndarray:
            rv.flags.writeable = False  # Weak immutability
        return rv


class Stats(object):

    def __getattr__(self, name):
        if hasattr(original_stats, name):
            # Pass through constructor arguments to StatsWrapper, including name of
            # the intended object
            return lambda *a, **kw: StatsWrapper(*a, **dict(kw, vname=name))
        raise AttributeError('scipy.stats has no attribute %s' % name)

rvs = Stats()


def seeded(stoch_func, state):
    return partial(stoch_func, prngstate=random_state(state))
