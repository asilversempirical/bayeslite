from hypothesis import given, strategies as st, settings, Verbosity, assume
from hypothesis import seed
from math import erf
import numpy as np

import bdb_test_distributions as btd
from gaussian import gaussian_log_pdf
from kl import kullback_leibler
import threshold


@st.composite
def normal_check_statistic_args(draw):
    # XXX: Crosscat breaks down if the mean is too large
    mean = draw(st.floats(min_value=-1e6, max_value=1e6))
    # XXX: Crosscat breaks down if the variance is too tiny
    sigma = draw(st.floats(min_value=1e-3, max_value=1e4))
    # XXX: https://github.com/probcomp/bayeslite/issues/388
    assume(abs(mean / (sigma**2)) < 1e5)
    return dict(
        sample_size=draw(st.integers(min_value=100, max_value=200)),
        mean=mean, sigma=sigma,
        numchains=draw(st.integers(min_value=1, max_value=10)),
        # Frequent failures occur here if min_value=5.
        numiterations=draw(st.integers(min_value=10, max_value=30)),
        seed=draw(st.randoms()))


def normal_check_statistic(sample_size, mean, sigma, numchains, numiterations,
                           seed):
    # Construct numpy RandomState from random.Random drawn from hypothesis
    r = np.random.RandomState([seed.randint(0, 2**32 - 1) for _ in xrange(8)])
    sample = r.normal(mean, sigma, sample_size)
    tbdb = btd.BayesDBTestDistribution(sample, r, numchains, numiterations)
    posterior_sample = tbdb.simulate(100, prngstate=r)
    real_pdf = gaussian_log_pdf(mean, sigma)
    kl = kullback_leibler(posterior_sample, tbdb.predictive_logpdf, real_pdf)
    # Return the minimum tail probability for 0, which should be the answer.
    rv = erf(abs(kl.estimate / kl.se)) / 2
    return rv


@seed(0)  # Same tests every time
# Pass `verbosity=Verbosity.verbose` to see what's going on
@settings(verbosity=Verbosity.normal,  # Verbosity.verbose for more info
          max_examples=10,
          # Don't work too hard to shrink failures.  Doesn't seem to help much.
          max_shrinks=1)
@given(st.data())
def test_normal(data):
    # Derived by
    # threshold.compute_sufficiently_stringent_threshold(
    #  lambda: normal_check_statistic(**normal_check_statistic_args().example),
    #  6, 1e-10, verbose=True)
    thresh = threshold.TestThreshold(threshold=0.0210347956630983,
                                     failprob=1.0005966255564656e-10,
                                     sample_size=5311)

    def check_statistic():
        return normal_check_statistic(**data.draw(
            normal_check_statistic_args()))
    threshold.check_generator(check_statistic, 6,
                              thresh.threshold, thresh.failprob)
