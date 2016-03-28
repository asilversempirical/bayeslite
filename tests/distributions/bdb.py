"""A wrapper around a bdb instance using the GPM interface.

Although the system wrapped up in a BayesDB instance is extremely protean, we
can make all the public methods here referentially transparent by doing all the
training during initialization from a fresh bdb.

BDB can be instantiated as either BDB(sample, seed, numchains, numiterations),
or BDB(bdb=<other BDB instance>). In the first case, a fresh bdb is created,
seeded according to seed, and populated with the sample, and trained on it on
numchains chains for numiterations iterations. In the second case, a copy is
taken of bdb, the copy's seed is set, and then its models are trained for a
further numiterations.

"""

from copy import deepcopy
from itertools import chain
import numpy as np

from distributions.distribution import TestDistribution
from util import bql, entropy

maxint = 2**63 - 2  # Largest integer random_integers can handle

# XXX: We need to make a CC engine with an explicit seed, here. Most of the
# entropy is generated in CC. See src/shell/src/main.py for an example of
# making a CC engine.

class BDB(TestDistribution):

    def __init__(self, sample=None, prngstate=None, numchains=None,
                 numiterations=None, bdb=None):
        if bdb is None:
            self.bdb = bql.bdb_open()
            self.seed(prngstate)
            self.sample = sample
            self._populate_model()
            if numchains is None:
                raise ValueError('Explicitly give number of chains')
            self._create_chains(numchains, prngstate)
            if numiterations is None:
                raise ValueError('Explicitly give number of iterations')
            self._analyze(numiterations, prngstate)
        else:
            self.bdb = deepcopy(bdb.bdb)
            if sample is not None:
                self.sample = np.array(list(chain(self.sample, sample)))
                self._populate_model()
            if numchains is not None:
                raise NotImplementedError(
                    "Don't make chains with different training lengths!")
            if numiterations is not None:
                self._analyze(numiterations, prngstate)

    def seed(self, prngstate):
        "Set the seeds for the bdb's RNGs"
        # Search on flowdock for "what's the role of weakprng in bayeslite?"
        # for discussion.  Currently weakprng's only role is to seed the python
        if isinstance(prngstate, entropy.HighEntropyRandomState):
            prngstate = prngstate.random_integers(-maxint, maxint)
        else:
            raise TypeError(
                'prngstate must be util.entropy.HighEntropyRandomState')
        self.bdb.set_entropy(prngstate)

    def forbidden(self, message):
        def forbidden(message):
            raise RuntimeError(message)
        return forbidden

    def _populate_model(self):
        "Put the given data into a table so that the bdb can operate on it"
        self.bdb.sql_execute('CREATE TABLE D (c float)')
        insertcmd = 'INSERT INTO D (c) VALUES'
        values = ', '.join('(%s)' % s for s in self.sample)
        self.bdb.sql_execute(insertcmd + ' ' + values)
        self._populate_model = self.forbidden('Already populated')

    def _create_chains(self, numchains, prngstate):
        "Create the model states the MCMC chains will iterate on."
        self.seed(prngstate)
        cmd = 'CREATE GENERATOR D_cc FOR D USING crosscat("c" numerical)"'
        self.bdb.execute(cmd)
        self.bdb.execute('INITIALIZE %i MODELS for D_cc' % numchains)
        self._create_chains = self.forbidden('Already created')

    def _analyze(self, numiterations, prngstate=None):
        "Run through numiterations MCMC steps."
        self.seed(prngstate)
        self.bdb.execute('ANALYZE D_cc FOR %i ITERATIONS WAIT' % numiterations)
        self._analyze = self.forbidden('Already analyzed')

    def simulate(self, samplesize, prngstate=None):
        "Draw samplesize variates from the estimated posterior distribution"
        self.seed(prngstate)
        simcmd = 'SIMULATE c from D_cc LIMIT %i' % samplesize
        return np.array(self.exec_to_array(simcmd))

    def make_predictive_logpdf(self, sample):
        "Return function f(x \in sample) => bdb-estimated logprob"
        probcmd = 'PREDICTIVE PROBABILITY of c FROM D_cc'
        probs = np.array(self.exec_to_array(probcmd))
        assert sample.shape == probs.shape
        self._predictive_logpdf = dict(zip(sample, np.log(probs))).__getitem__

    def predictive_logpdf(self, x):
        if not hasattr(self, '_predictive_logpdf'):
            self.make_predictive_logpdf(x)
        return self._predictive_logpdf
