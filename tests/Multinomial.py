from __future__ import division
from collections import namedtuple as nt
import itertools as it
import numpy as np

from reals import ispositive, isfinite

TMSnt = nt('TruncatedMultinomialSample', 'multinomial sample')


class TruncatedMultinomialSample(TMSnt):

    """Return value for Multinomial.truncated_sample

    `multinomial`: the reweighted weights for the observed events
    `sample`: the sample drawn

    """

    def __init__(self, multinomial, sample):
        if not isinstance(multinomial, Multinomial):
            raise ValueError('multinomial must be a Multinomial')
        indices = set(sample)
        all_witnessed = (len(indices) == len(multinomial.weights))
        if (indices != set(xrange(len(indices)))) or (not all_witnessed):
            raise ValueError(
                'Sample should have been reindexed so that all indices occur')


class Multinomial(nt('Multinomial', 'weights')):

    def __init__(self, weights):
        if not all(ispositive(w) for w in weights):
            raise ValueError('weights should be sequence of positive reals')
        self.cdf = np.cumsum(weights)
        if not np.isclose(self.cdf[-1], 1):
            raise ValueError('weights must sum to 1')
        self.lpdf = np.log(weights)

    def sample(self, prngstate, n=1):
        return self.cdf.searchsorted(prngstate.uniform(0, 1, n))

    def truncated_sample(self, prngstate, n):
        """Draw `n` observations, reindex events so that those which occurred in the
        sample are first, drop other indices, and return the reindexed sample
        and the corresponding reweighted multinomial.

        """
        counts = prngstate.multinomial(n, self.weights)
        nonzero = counts.nonzero()
        weights = self.weights[nonzero]
        multinomial = Multinomial(weights / weights.sum())
        sample = it.chain(*(c * [i] for i, c in enumerate(counts[nonzero])))
        sample = prngstate.permutation(list(sample))
        return TruncatedMultinomialSample(multinomial, sample)

    def lpdf(self, x):
        return self.lpdf[x]

    def logmean(self, v):
        "log(exp(v)*self.weights)"
        lprods = self.lpdf + v
        minval = lprods.min()
        rv = np.log(np.exp(lprods - minval).sum()) + minval
        if not isfinite(rv):
            raise RuntimeError('under/overflow')
        return rv


class DirichletProcess(nt('DirichletProcess', 'alpha')):

    def __init__(self, alpha):
        if not ispositive(alpha):
            raise ValueError('alpha must be positive')

    def sample(self, prngstate):
        remaining = 1.
        rv = []
        while remaining > 1e-20:
            wi = prngstate.dirichlet([1, self.alpha])[0]
            length = wi * remaining
            rv.append(length)
            remaining -= length
        total = sum(rv)
        assert (min(rv) > 0) and np.isclose(total, 1)
        return Multinomial(np.array(rv) / total)

    def lpdf(self, m):
        raise NotImplementedError
