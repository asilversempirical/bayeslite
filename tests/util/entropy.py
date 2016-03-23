"""Wrap random variates to enforce explicit settings of PRNG seeds.

To use a distribution from scipy.stats, refer to it via rvs, e.g.,
n=rvs.norm(0,1) will return a normal distribution.

"""

import functools
from numpy.random import RandomState
from numpy import ndarray
from scipy import stats as original_stats

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
        if not isinstance(random_state, RandomState):
            raise ValueError('Need to explicitly set random_state with '
                             'numpy.random.RandomState instance')
        # standard rvs method mutates a copy of the kwds dictionary...
        rv = self.__obj.rvs(size=size, random_state=random_state)
        if type(rv) == ndarray:
            rv.flags.writeable = False  # Weak immutability
        return rv


class Stats(object):

    def __getattr__(self, name):
        # Pass through constructor arguments to StatsWrapper, including name of
        # the intended object
        return lambda *a, **kw: StatsWrapper(*a, **dict(kw, vname=name))

rvs = Stats()


def random_state(state):
    if isinstance(state, int):
        state = RandomState(state)
    if not isinstance(state, RandomState):
        raise ValueError('Random seed should be int or RandomState instance')
    return state


def seeded(stoch_func, state):
    return functools.partial(stoch_func, prngstate=random_state(state))
