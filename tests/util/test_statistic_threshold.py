#!/usr/bin/env python

import itertools
import math
import numbers
import random

SMYTHE_THOMPSON_NUMBER = 17

def nones(n):
    "Returns an iterator which generates None n times"
    return itertools.repeat(None, n)

def lbeta(m,n):
    """Return log(Beta(m,n))"""
    return math.lgamma(m)+math.lgamma(n)-math.lgamma(m+n)

def failprob_threshold(observed, ns, threshold):
    """Takes ([float] observed, int ns, float threshold)

    observed: iid numeric samples of some test statistic,
    ns: number of iid samples to be drawn when the test statistic is being used
        in an integration test,
    threshold: desired maximum probability that "ns" iid samples from
    the same underlying distribution "observed" was drawn from are all less
    than some bound "x".

    Returns (float p, float x). Return value "p" is the posterior probability
    of drawing "ns" samples from the underlying distribution which are all less
    than "x" is less than "threshold". "x" is the ML estimate of the
    threshold**(1/ns) quantile"""

    # Type checking
    if not all(isinstance(d, numbers.Number) for d in observed):
        raise ValueError, 'observed is not a list of numeric values'
    if (threshold > 1) or (threshold < 0):
        raise ValueError, 'threshold is not a probability'
    if (round(ns) != ns) or (ns < 1):
        raise ValueError, 'ns is not a natural number'

    # Compute the quantile which should be tested for in each subtest
    observed = sorted(observed)
    sub_threshold = threshold ** (1./ns)
    mlxidx = int(len(observed)*sub_threshold)
    mlx = observed[mlxidx]
    if observed.count(mlx) > 1:
        # If mlx occurs more than once, it's likely that it contains
        # non-trivial probability mass. If the target quantile lies within that
        # mass then the key assumption of this approach (i.e. that P(y<mlx) is
        # approximately sub_threshold) has broken down and a different
        # threshold should be chosen.
        raise ValueError, 'Requested quantile may lie in Dirac delta fn'

    # Compute the observed counts below and above the threshold mlx
    below, above = max(0,  mlxidx-1), len(observed) - mlxidx - 1

    # We have observed "below" samples less than or equal to mlx, "above"
    # samples above it. If we treat these as observations of a binomial, the
    # posterior on P(y<mlx) is a Beta(below+1,above+1) distribution, call it
    # PB. The posterior probability of "ns" iid samples less than or equal to
    # mlx is the integral over the unit interval of (q**ns)*PB(q,1-q), which is
    # the integrand of Beta(below+ns+1,above+1), i.e.
    probfail = math.exp(lbeta(below+ns+1,above+1) - lbeta(below+1,above+1))
    return probfail, mlx

def test_failprob_threshold():
    random.seed(SMYTHE_THOMPSON_NUMBER)
    def sample(n):
        return [random.gauss(0, 1) for _ in nones(n)]
    target_prob, test_sample_size = 1e-2, 6
    prob, thresh = failprob_threshold(sample(10000), test_sample_size,
                                      target_prob)
    samples = [all(v < thresh for v in sample(test_sample_size))
               for _ in nones(int(100/target_prob))]
    assert samples.count(True) < 200

def compute_sufficiently_stringent_threshold(generator, ns, threshold):
    """generator is a function which takes no arguments and returns a float. Its
    return values are assumed to be iid. ns and threshold are as in
    sufficiently_stringent_p. Returns a float x such that the probability of
    drawing ns samples from generator which are all less than x is less than
    threshold, a float probfail which is the actual estimated probability of ns
    samples less than x, and an int which is the number of samples drawn to
    make the estimate."""
    batchsize = int(threshold**(-1./ns)) + 1
    observed = []
    while True:
        observed.extend(generator() for _ in nones(batchsize))
        probfail, x = failprob_threshold(observed, ns, 0.9*threshold)
        if probfail < threshold:
            return x, probfail, len(observed)

class MultipleTestStatisticFailures(RuntimeError):

    """Raised when a test statistic is too low too many times"""

    def __init__(self, generator, ns, threshold, statistics):
        self.generator, self.ns, self.threshold = generator, ns, threshold
        self.statistics = statistics

def test_generator(generator, ns, threshold, probfail):
    statistics = []
    for numfailures in range(ns):
        statistics.append(generator())
        if statistics[-1] >= threshold:
            return numfailures
    raise MultipleTestStatisticFailures(generator, ns, threshold, statistics), \
        ('%s has been less than %.3g %i times, the probability of which was '
         ' estimated to happen one time in %.3g') % (
             generator, threshold, ns, 1/probfail)

def main():
    import argparse
    from importlib import import_module
    parser = argparse.ArgumentParser(
        description='Determine threshold for statistical test')
    parser.add_argument('generator', metavar='generator',
        help='Fully-qualified module + function name for test-statistic function.  '
        'E.g. numpy.random.standard_normal')
    parser.add_argument('num_iterations', metavar='num_iterations',
                        help='The number of times generator will be allowed to '
                        'return a value less than threshold before failure is '
                        'reported')
    parser.add_argument('maxfailprob', metavar='maxfailprob',
                        help='The maximum probability of a failure report')
    args = parser.parse_args()
    genname = args.generator.split('.')
    modname = '.'.join(genname[:-1])
    generator = getattr(import_module(modname), genname[-1])
    ns = int(args.num_iterations)
    threshold = float(args.maxfailprob)
    x, probfail, samplesize = compute_sufficiently_stringent_threshold(
        generator, ns, threshold)
    print(('Upper bound of %.3f is estimated to have probability %.3g < %.3g '
           'of being met %i times in a row, based on a sample of size %i') % (
               x, probfail, threshold, ns, samplesize))

if __name__ == '__main__':
    main()
