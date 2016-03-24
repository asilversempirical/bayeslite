"""Model of normal distribution and normal-inverse-gamma prior."""

from numbers import Number
from numpy import sqrt, pi, log
import scipy
from scipy.special import gammaln
from util.entropy import rvs

from distributions.distribution import TestDistribution


class Normal(TestDistribution):

    """Normal distribution."""

    def __init__(self, mu, var):
        self.mu, self.var = mu, var
        self.n = rvs.norm(mu, sqrt(var))

    def simulate(self, n=1, prngstate=None):
        return self.n.rvs(n, random_state=prngstate)

    def logpdf(self, x):
        return self.n.logpdf(x)

    def __getstate__(self):
        return self.n.mean(), self.n.var()

    def __setstate__(self, state):
        self.__init__(*state)


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
        mus = [self.mudist(s).simulate(prngstate=prngstate) for s in sigmas]
        return [Normal(mu, sigma) for mu, sigma in zip(mus, sigmas)]

    def logpdf(self, n):
        if not isinstance(n, Normal):
            ValueError('Argument should be Normal instance')
        mu, sigma = n.mu, n.var
        return self.sigma.logpdf(sigma) + self.mudist(sigma).logpdf(mu)

    def posterior(self, data):
        """See Lemma 12, Jordan's "The Conjugate Prior for the Normal Distribution"
        http://www.cs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf
        (which is in terms of precision, not variance, hence the Inverse Gamma
        used here.)"""
        data = scipy.array(data).flatten()
        n0 = self.n0 + len(data)
        total = self.mumu * self.n0 + sum(data)
        # mean of "mu | tau, x" in Jordan's formula is mean of total data
        mumu = total / n0
        # "alpha +code, it' n/2" from Jordan's formula
        sigshape = self.sigshape + len(data) / 2.
        mean = data.mean()
        zeroed = data - mean
        t2 = sum(zeroed**2) / 2
        meandiffs = (mean - self.mumu)**2
        t3_denom = len(data) * self.n0 * meandiffs
        # "beta+0.5*sum(xi-x)**2+n*n0*(xbar-mu0)/2(n+n0)" from Jordan formula.
        sigrate = self.sigrate + t2 + t3_denom / (2 * n0)
        return NormalInvGammaPrior(mumu, sigshape, sigrate, n0)

    def predictive_logpdf(self, x):
        x = [x] if isinstance(x, Number) else x
        posterior = self.posterior(x)

        def beta_factor(p):
            return p.sigshape * log(p.sigrate)
        # Eq. 99 of http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        return (gammaln(posterior.sigshape) - gammaln(self.sigshape) +
                beta_factor(self) - beta_factor(posterior) +
                log(self.n0 / posterior.n0) / 2 - len(x) * log(2 * pi) / 2.)

    def __getstate__(self):
        # FIXME: This is dangerous... Is there a way to read this out of the
        # __init__ signature, e.g. with inspect.getargspec? Could have a test
        # which roundtrips every TestDistribution?
        return self.mumu, self.sigshape, self.sigrate, self.n0

    def __setstate__(self, state):
        self.__init__(*state)
