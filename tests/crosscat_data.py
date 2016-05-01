from __future__ import division
from functools import partial
from collections import namedtuple as nt
import itertools as it
import numpy as np

import composite_distributions; reload(composite_distributions)
from composite_distributions import IndependentVariables, Mixture
from composite_distributions import MixtureSample
# import Multinomial; reload(Multinomial)
from Multinomial import DirichletProcess
# import Normal; reload(Normal)
from Normal import NormalInverseGammaPrior
from safezip import safezip


def col_hyperpriors(prngstate):
    # Other values must be small for there to be interesting variation in the
    # table data. And numpy.random.gamma can't handle really tiny shape
    # parameters: it just returns 0.  So bound it away from 0.
    tiny = partial(prngstate.uniform, 1e-2, 0.1)
    return NormalInverseGammaPrior(mu=prngstate.uniform(-1e6, 1e6),
                                   n0=tiny(), sigrate=tiny(), sigshape=tiny())

CCDnt = nt('CrossCatData', 'viewprior rpriors hyperpriors viewmap sample')


class CrossCatData(CCDnt):

    def __init__(self, viewprior, rpriors, hyperpriors, viewmap, sample):
        nrows = set(len(m.assignments) for m in sample)
        if len(nrows) != 1:
            raise ValueError('sample should be for a fixed number of rows')
        self.nrows = nrows.pop()
        colidxs = set(it.chain(*viewmap))
        self.ncols = max(colidxs) + 1
        if colidxs != set(xrange(self.ncols)):
            raise ValueError('viewmap should be a map from views to columns')
        if not isinstance(viewprior, DirichletProcess):
            raise ValueError('viewprior should be a `DirichletProcess`')
        if not all(isinstance(rp, DirichletProcess) for rp in rpriors):
            raise ValueError('rpriors should be list of `DirichletProcess`es')
        if not all(isinstance(hp, NormalInverseGammaPrior)
                   for hp in hyperpriors):
            raise ValueError('hyperpriors should be a list of '
                             '`NormalInverseGammaPrior`s')
        if len(hyperpriors) != self.ncols:
            raise ValueError('There should be a hyperprior per column')
        if len(set(map(len, [rpriors, viewmap, sample]))) != 1:
            raise ValueError(
                'rpriors, viewmap, sample should map to views')
        if not all(isinstance(samp, MixtureSample) for samp in sample):
            raise ValueError('sample should be a list of `MixtureSample`s')
        r = RandomState(0)
        if not all([s.mixture.sample(r, n=1).values[0].shape == (len(vm),)
                    for s, vm in safezip(sample, viewmap)]):
            raise ValueError('sample mixtures should generate same shape as '
                             'corresponding views')

    def as_array(self):
        rows = np.zeros((self.nrows, self.ncols))
        rows.fill(np.nan)
        for vidx, cols in enumerate(self.viewmap):
            for colidx, col in enumerate(cols):
                rows[:, col] = self.sample[vidx].values[:, colidx]
        assert not any(np.isnan(rows.flatten())), 'Should fill in whole table'
        return rows


def crosscat_data(nrows, ncols, prngstate):
    # p. 8, step 1.
    view_prior = DirichletProcess(prngstate.gamma(1))
    # p. 8, step 2(a).
    per_col_hyperpriors = [col_hyperpriors(prngstate) for _ in xrange(ncols)]
    # p. 8, step 2(b).
    view_dist = view_prior.sample(prngstate)
    col_views = view_dist.truncated_sample(prngstate, ncols)
    # Make a map from view indices to lists of columns assigned to those views
    viewmap = [[] for _ in xrange(max(col_views.sample) + 1)]
    for colidx, viewidx in enumerate(col_views.sample):
        viewmap[viewidx].append(colidx)
    # p. 9, step 3(a).  The CRP for generating categories in each view
    rpriors = [DirichletProcess(a)
               for a in prngstate.gamma(1, size=len(viewmap))]
    # p. 9, step 3(b).  The assignment of rows to categories per the CRPs
    row_assignments = [p.sample(prngstate).truncated_sample(prngstate, nrows)
                       for p in rpriors]
    # p. 9, step 3(c). First, get the view distributions as mixtures. We won't
    # actually sample directly from these mixtures, since we already have
    # assignments of rows to mixture components from row_assignments. But they
    # clarify what's going on here, and will make it easier to test crosscat's
    # output.

    # The independent per-column components per category, per view
    vcomps = [[[per_col_hyperpriors[ci].sample(prngstate)[0] for ci in vmap]
               # Components of mixtures correspond to weights of multinomials
               for _ in row_assignment.multinomial.weights]
              for vmap, row_assignment in safezip(viewmap, row_assignments)]
    # Combine the column component dists as independent vars per view
    vcomps = [[IndependentVariables(comp) for comp in view] for view in vcomps]
    cat_dists = [Mixture(weights=asst.multinomial, components=vcomp)
                 for asst, vcomp in safezip(row_assignments, vcomps)]
    # Now sample for each view
    cols = [np.array([m.components[c].sample(prngstate, 1)[0]
                      for c in a.sample])
            for m, a in safezip(cat_dists, row_assignments)]
    vsamps = [MixtureSample(mixture=m, assignments=a.sample, values=c)
              for m, a, c in safezip(cat_dists, row_assignments, cols)]
    return CrossCatData(viewprior=view_prior, rpriors=rpriors,
                        hyperpriors=per_col_hyperpriors, viewmap=viewmap,
                        sample=vsamps)


from numpy.random import RandomState
r = RandomState(5)
t = crosscat_data(1000, 1000, r).as_array()
