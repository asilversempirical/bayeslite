from cPickle import dumps, loads
from scipy.stats import norm

from distributions.distribution import TestDistribution
from tests.test_immutable import test_immutable, ImmutableTest


class T(TestDistribution, ImmutableTest):
    pass


class TR(ImmutableTest, TestDistribution):
    pass


def test_immutability():
    test_immutable(T)
    test_immutable(TR)  # Does MRO matter??


class TestThreadEnforcement(TestDistribution):

    """Test class for prng threading enforcement."""

    def __init__(self):
        self.norm = norm(0, 1)

    def simulate(self, numsamples=1, prngstate=None):
        """Example sample method."""
        return self.norm.rvs(numsamples, random_state=prngstate)


def test_threading_enforcement():
    """Check that threading enforcement works."""
    test_instance = TestThreadEnforcement()
    try:
        test_instance.simulate()
    except ValueError, exception:
        assert len(exception.args) == 1
        assert exception.args[0].startswith(
            'Need to explicitly specify prngstate in call to function '
            'simulate')
    else:
        raise RuntimeError('Failed to raise error about prngstate')
