"""Test code for normal distribution."""

from numbers import Number
from numpy import sqrt, pi, log, abs
import scipy
from scipy.special import gammaln
from util.entropy import rvs, seeded, random_state

from test_distribution import TestDistribution
from util import test_statistic_threshold as tst
from util.test_arithmetic import safe_mean, safe_product, safe_divide, fsum
from util.test_math import kullback_leibler


class Normal(TestDistribution):

    """Normal distribution."""

    def __init__(self, mu, var):
        self.n = rvs.norm(mu, sqrt(var))

    def simulate(self, n=1, prngstate=None):
        return self.n.rvs(n, random_state=prngstate)

    def logpdf(self, x):
        return self.n.logpdf(x)


class NormalInvGammaPrior(TestDistribution):

    """Prior distribution for a normal with variance uncertainty modeled by
    sigma ~ InvGamma(sigshape, sigrate), mean uncertainty modeled by N(mumu,
    sigma/n0), assuming n0 observations so far."""

    def __init__(self, mumu, sigshape, sigrate, n0=1):
        if min(sigshape, sigrate) <= 0:
            raise ValueError('sigshape & sigrate must be positive')
        if not isinstance(n0, Number) or n0 != round(n0) or n0 < 1:
            raise ValueError('n0 must be natural number')
        self.sigma = rvs.invgamma(sigshape, scale=sigrate)
        # It seems these can't be read out of an invgamma
        self.sigshape, self.sigrate = sigshape, sigrate
        self.mumu, self.n0 = mumu, float(n0)

    def mudist(self, sigma, prngstate=None):
        return Normal(self.mumu, sigma / float(self.n0))

    def simulate(self, n=1, prngstate=None):
        sigmas = self.sigma.rvs(n, random_state=prngstate)
        mus = [self.mudist(s).rvs(n, random_state=prngstate) for s in sigmas]
        return [rvs.norm(mu, sqrt(sigma), random_state=prngstate)
                for mu, sigma in zip(mus, sigmas)]

    def logpdf(self, n):
        if n.dist != scipy.stats.norm(0, 1).dist:
            ValueError('Argument should be scipy.stats.norm instance')
            mu, sigma = n.mean(), n.var()
        return self.sigma.logpdf(sigma) + self.mudist(sigma).logpdf(mu)

    def posterior(self, data):
        """See Lemma 12, Jordan's "The Conjugate Prior for the Normal Distribution"
        http://www.cs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf
        (which is in terms of precision, not variance, hence the Inverse Gamma
        used here.)"""
        data = scipy.array(data).flatten()
        n0 = self.n0 + len(data)
        total = fsum([safe_product([self.mumu, self.n0]), fsum(data)])
        # mean of "mu | tau, x" in Jordan's formula is mean of total data
        mumu = safe_divide(total, n0)
        # "alpha + n/2" from Jordan's formula
        sigshape = fsum([self.sigshape, len(data) / 2.])
        mean = safe_mean(data)
        zeroed = scipy.array([fsum([d, -mean]) for d in data])
        t2 = safe_divide(fsum(zeroed**2), 2)
        meandiffs = scipy.array(fsum([mean, -self.mumu]))**2
        t3_denom = safe_product([len(data), self.n0, meandiffs])
        # "beta+0.5*sum(xi-x)**2+n*n0*(xbar-mu0)/2(n+n0)" from Jordan formula.
        sigrate = fsum([self.sigrate, t2, safe_divide(t3_denom, 2 * n0)])
        return NormalInvGammaPrior(mumu, sigshape, sigrate, n0)

    def predictive_logpdf(self, x):
        x = [x] if isinstance(x, Number) else x
        posterior = self.posterior(x)

        def beta_factor(p):
            return p.sigshape * log(p.sigrate)
        # Eq. 99 of http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        return fsum([gammaln(posterior.sigshape), -gammaln(self.sigshape),
                     beta_factor(self), -beta_factor(posterior),
                     log(self.n0 / posterior.n0) / 2,
                     -len(x) * log(2 * pi) / 2.])


def normal_prior_check_statistic(prngstate):
    """Test statistic which sanity checks NormalInvGammaPrior by verifying that
    the actual mean & variance from which a sample is drawn lie in the likely
    region of the posterior given the sample."""
    mu, var, samplesize = 5, 2, 100
    n = Normal(mu, var)
    sample = n.simulate(samplesize, prngstate=prngstate)
    prior = NormalInvGammaPrior(mu, 1. / var, 1, 1)
    posterior = prior.posterior(sample)
    return min(posterior.sigma.sf(var),
               posterior.mudist(var, prngstate).n.sf(mu))

ntest_statistic = seeded(normal_prior_check_statistic, 17)
# tst.compute_sufficiently_stringent_threshold(ntest_statistic, 6, 1e-15)
# => (0.0013068798846691543, 9.628012077249724e-16, 42478)
ntest_thresh = 1.30688e-3


def test_normal_prior():
    tst.check_generator(ntest_statistic, 6, ntest_thresh, 1e-15)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')
import numpy as np


def kullback_leibler_normal():
    """Display the convergence of the posterior of a NormalInvGammaPrior as it gets
    increasingly large samples from a given normal distribution. Not a very
    clear visualization for others -- too much information, I suppose.

    Since all of the pdfs are roughly normal, it would be better to display
    summary statistics of them: the pdf values of actual mean and variance
    seems like a reasonable thing...

    """
    kl_divs = []
    prngstate = random_state(17)
    mu, var = 0, 1
    n = Normal(mu, var)
    sample = list(n.simulate(1, prngstate=prngstate))
    number_doublings = 8
    samplesizes = 2**scipy.arange(0, number_doublings, 1)
    lowbound, highbound = -5, 5
    x = np.linspace(lowbound, highbound, 100)
    X, Y = np.meshgrid(x, log(samplesizes) / log(2))
    Z = scipy.zeros(X.shape)
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the pdf of the actual distribution, at the back of the image
    ax.plot(x, len(x) * [number_doublings],
            np.exp(map(n.logpdf, x)), lw=5, color='r',
            label='Actual distribution')
    base = 1000000000
    ax.set_ylabel('Slices are posterior distributions given sample of specified size')
    ax.set_xlabel('Probability space (X)')
    ax.set_zlabel('Probability density at X / $\log_{%i}$(|KL divergence|)' % base)
    for sidx, samplesize in enumerate(samplesizes):
        assert len(sample) == samplesize
        prior = NormalInvGammaPrior(mu, 1. / var, 1, 1)
        posterior = prior.posterior(sample).predictive_logpdf
        kl_divs.append(kullback_leibler(sample, posterior, n.logpdf)[0])
        Z[sidx, :] = np.exp(map(posterior, x))
        sample.extend(list(n.simulate(len(sample), prngstate=prngstate)))
    ax.plot_wireframe(X, Y, Z)
    ax.set_yticklabels(['$2^{%i}$' % s for s in ax.get_yticks()])
    ax.plot(number_doublings*[lowbound-1], xrange(number_doublings),
            -log(abs(kl_divs))/log(base), lw=5, color='b',
            label='$-\log_{%i}$(|KL divergence|)' % base)
    plt.legend()
    plt.title('Convergence of NormalInvGammaPrior with increasing sample size')
    plt.ion()
    plt.show()
    return ax
