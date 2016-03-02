"""Base class for test GPMs"""

from functools import wraps
from numpy.random import RandomState

from gpmcc.dists.distribution import DistributionGpm

from util.immutable import Immutable

class TestDistribution(Immutable, DistributionGpm):

    "Base distribution for test classes"

    # The number of samples to provide as observations during a test
    numobservations = 100

    # The number of iterations each chain should be put through, before drawing
    # from the posterior sample during a test
    numiterations = 100

    # Methods which should be checked for prngstate
    stochastic_methods = '''simulate transition_params'''.split()

    def __getattribute__(self, name):
        "Enforce threading of prngstate"
        attr = super(TestDistribution, self).__getattribute__(name)
        if name in TestDistribution.stochastic_methods:
            return self.stochastic_wrapper(attr)
        return attr

    @staticmethod
    def stochastic_wrapper(func):
        "Enforce threading of prngstate"

        @wraps(func)
        def wrapped(*args, **kwrds):
            'Check for prngstate before calling intended function'
            if ('prngstate' not in kwrds) or \
               (not isinstance(kwrds['prngstate'], (int, RandomState))):
                raise ValueError(
                    ('Need to explicitly specify prngstate in call to '
                     'function %s ') % func.__name__)
            return func(*args, **kwrds)

        return wrapped
<<<<<<< variant A

    @staticmethod
    def init_wrapper(self, ):
        pass
>>>>>>> variant B
======= end
