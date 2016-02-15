import scipy
import scipy.integrate
from scipy.stats import norm
from numpy import log, sqrt, exp, inf

from util import test_statistic_threshold as tst


def kullback_leibler(postsample, postpdf, comppdf):
    """Estimate KL-divergence of sample (a collection of values) w.r.t. known pdf,
    comp_pdf, which returns the density when passed a sample. Return value is
    (estimated_kl, variance), where variance is the estimated sampling variance
    of estimated_kl. post_sample is an approximate sample from the distribution
    approximately represented by post_pdf

    """
    klsamples = log(map(postpdf, postsample)) - log(map(comppdf, postsample))
    # In general the mean and variance of log(P(x)) is not known to be finite,
    # but it will be for any distribution crosscat generates at the moment,
    # because they all have finite entropy.  Hence CLT applies
    return klsamples.mean(), klsamples.var() / sqrt(len(klsamples))


def kullback_leibler_numerical(lpdf1, lpdf2):
    klf = lambda x: exp(lpdf1(x)) * (lpdf1(x) - lpdf2(x))
    return scsipy.integrate.quad(klf, -inf, inf)


def kullback_leibler_test_statistic():
    dist1, dist2 = norm(0, 1), norm(0, 2)
    lpdf1, lpdf2 = dist1.logpdf, dist2.logpdf
    pdf1, pdf2 = dist1.pdf, dist2.pdf
    exact, error = kullback_leibler_numerical(lpdf1, lpdf2)
    estimate, std = kullback_leibler(dist1.rvs(100), pdf1, pdf2)
    return norm(estimate, std).sf(exact)

# Ran tst.compute_sufficiently_stringent_threshold(
# kullback_leibler_test_statistic, 6, 1e-20), stopped it after 32399 runs, at
# which time failprob_threshold reported that 4.516e-6 was an acceptable
# threshold, for a failure probability of 1.7e-20.


def test_kullback_leibler():
    tst.test_generator(kullback_leibler_test_statistic, 6, 4.516e-6, 1.7e-20)


def graph_kullback_leibler_test_statistic():
    from matplotlib import pyplot as plt
    plt.hist([kullback_leibler_test_statistic() for _ in xrange(100)])
