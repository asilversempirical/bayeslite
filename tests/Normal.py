from __future__ import division
from collections import namedtuple as nt
import numpy as np

from reals import ispositive as pos, isreal


class Normal(nt('Normal', 'mu sigma')):

    neglogroottwopi = -np.log(2 * np.pi) / 2

    def __init__(self, mu, sigma):
        if not (isreal(mu) and pos(sigma)):
            raise ValueError('mu and sigma must be real numbers')
        if sigma <= 0:
            raise ValueError('sigma must be positive')
        self.lnorm = -np.log(sigma) - self.neglogroottwopi

    def sample(self, prngstate, n=1):
        return prngstate.normal(self.mu, self.sigma, n)

    def lpdf(self, x):
        return self.lnorm - ((x - self.mu) / self.sigma / 2)**2

NIGPrior = nt('NormalInverseGammaPrior', 'mu n0 sigrate sigshape')


class NormalInverseGammaPrior(NIGPrior):

    """Prior on normals

    `mu`: mean of observations
    `n0`: effective number of observations
    `sigrate`: half of number of degrees of freedom
    `sigshape`: half of sum of squares of observations

    See  Jordan's notes, p. 6, for formulae:
    http://www.cs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf

    """
    def __init__(self, mu, n0, sigrate, sigshape):
        if not (isreal(mu) and pos(n0) and pos(sigrate) and pos(sigshape)):
            raise ValueError('mu must be real, other args must be positive')

    def sample(self, prngstate, n=1):
        # \tau ~ \Gamma(sigshape, sigrate)
        taus = prngstate.gamma(self.sigshape, 1 / self.sigrate, n)
        # Prevented by the lower bound in tiny in crosscat_data.col_hyperpriors
        assert taus.min() > 0, 'This should never happen, but it does'
        # \sigma = 1/\sqrt(\tau)
        sigmas = taus**(-0.5)
        assert not any(np.isnan(sigmas) | np.isinf(sigmas))
        mus = prngstate.normal(self.mu, sigmas / np.sqrt(self.n0), n)
        return map(Normal, mus, sigmas)

    def lpdf(self, x):
        raise NotImplementedError('Unnecessary for now')
