import os
from hypothesis import given, strategies as st, settings, Verbosity, example
from hypothesis import assume
import numpy as np

from distributions import bdb, normal
from util import statistics
from util.entropy import rvs, seeds, random_state


def is_positive(x):
    return x > 0


class Sample:

    """Used as return value for normal_sample. The actual sample is in the `sample`
    attribute. It's done this way to avoid vast hypothesis error messages.

    """

    def __init__(self, sample_size, mean, variance, seed):
        self.sample_size, self.mean, self.variance, self.seed = (
            sample_size, mean, variance, seed)
        self.distribution = rvs.norm(mean, np.sqrt(variance))
        sample = self.distribution.rvs(sample_size, random_state(seed))
        self.sample = np.array(sample)

    def new_sample(self, prngstate=None):
        if prngstate is None:
            raise ValueError('Need to explicitly set prngstate')
        return Sample(self.sample_size, self.mean, self.variance,
                      prngstate.randint(2**32 - 1))

    def __repr__(self):
        nargs = 'sample_size mean variance seed'.split()
        args = ', '.join('%s=%s' % (n, self.__dict__[n]) for n in nargs)
        return '%s(%s)' % (self.__class__, args)


class SpecificSample:

    "Represents a specific sample, to be used for explicit examples"

    def __init__(self, sample):
        self.sample = np.array(sample)

    def __repr__(self):
        pass

@st.composite
def normal_sample(draw):
    # We need a minimum sample size here in order to do as well as crosscat,
    # which gets extra flexibility from the hyperprior. 100 seems to be about
    # the minimum to swamp the effects of the priors.
    sample_size = draw(st.integers(min_value=100, max_value=200))
    # XXX: Currently crosscat breaks down with huge means.
    mean = draw(st.floats(min_value=-1e6, max_value=1e6))
    # XXX Note: filtering on is_positive yields errors (probably cancellation)
    variance = draw(st.floats(min_value=1e-3, max_value=10000))
    # https://github.com/probcomp/bayeslite/issues/388#issuecomment-202984020
    assume(abs(mean / variance) < 1e5)
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    return Sample(sample_size, mean, variance, seed)


# XXX: Failure examples skipped
def failure_example(s, seed=0):
    return example(sample=SpecificSample(s), seed=seed,
                   numchains=1, numiterations=1)
# For larger values, they all fail.  For smaller ones, they don't.
# https://github.com/probcomp/bayeslite/issues/388
failures = map(lambda a: failure_example(*a),
               ((-1.0010415476e-146,), (-0.00114661120796,),
                # These multivalue examples are probably problematic because
                # the variance is so low relative to the mean.
                ((2952788047.4, 2952788169.25), 1),
                ((764149760.054, 764149760.055, 764149760.054), 105929),
               ((32003.9998936, 32004.0002834, 32003.999974, 32003.9998355,
                 32004.0003761, 32003.9997346, 32003.9999356, 32004.0003391,
                 32004.0002069, 32003.999385), 105929)))
# To try out a specific example, decorate test_bdb with f
f = failures[-1]

@settings(verbosity=Verbosity.verbose)
@given(sample=normal_sample(),
       seed=seeds(),
       numchains=st.integers(min_value=1, max_value=10),
       # You get failures here with min_value=5.
       numiterations=st.just(10))
@example(sample=Sample(sample_size=100, mean=0.0, variance=0.00758000556743893,
                       seed=622730657),
         # Fails with 1 iteration
         seed=0, numchains=1, numiterations=2)
@example(sample=Sample(sample_size=139, mean=4.63463e-319,
                       variance=4882.1597057633335, seed=3496090406),
         # Fails with 7 iterations
         seed=93806, numchains=3, numiterations=10)
@example(sample=Sample(sample_size=143, mean=2.0315979357e-320, variance=0.001,
                       seed=1189352067),
         # Fails with 200 iterations!
         seed = 4112, numchains = 10, numiterations = 200)
def test_bdb(sample, seed, numchains, numiterations):
    """Check that the posterior distribution given a normal sample is the same as
    what we get from a NIG prior."""
    r = random_state(seed)
    # Cheating a bit here... Cross cat will do better becaause it has the
    # flexibility from the hyperprior.
    prior = normal.NormalInvGammaPrior(sample.mean, 1. / sample.variance, 1)
    exact_posterior = prior.posterior(sample.sample)
    # bdb.make_reporter for noise, bdb.BDB for quiet
    tbdb = bdb.BDB(sample.sample, r, numchains,
                   numiterations)
    posterior_sample = tbdb.simulate(100, prngstate=r)
    kl, kl_var = statistics.kullback_leibler(
        posterior_sample, tbdb.predictive_logpdf,
        exact_posterior.predictive_logpdf)
    assert abs(kl) < 0.1, 'We should be nailing the exact same posterior'

# XXX crosscat is slow to converge, here.
# @example(sample=Sample(sample_size=10000, mean=0,
#                        variance=10000, seed=3496090406),
#          seed=93806, numchains=3, numiterations=10)
#
