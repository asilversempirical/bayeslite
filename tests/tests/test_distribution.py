from scipy.stats import norm

from distributions.distribution import TestDistribution


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

def main():
    test_threading_enforcement()
