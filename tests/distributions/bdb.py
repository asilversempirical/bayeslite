"""A wrapper around a bdb instance using the GPM interface.

Although the system wrapped up in a BayesDB instance is extremely protean, we
can make all the public methods here referentially transparent by doing all the
training during initialization from a fresh bdb. Note, however, that at the
moment these methods are not thread-safe: The BayesDB instance is seeded at the
start of each method, but no lock is taken.

BDB can be instantiated as either BDB(sample, seed, numchains, numiterations),
or BDB(bdb=<other BDB instance>). In the first case, a fresh bdb is created,
seeded according to seed, and populated with the sample, and trained on it on
numchains chains for numiterations iterations. In the second case, a copy is
taken of bdb, the copy's seed is set, and then its models are trained for a
further numiterations.

If you want to make a script for reporting a failure, replace the invocation of
BDB with make_reporter.

"""

from copy import deepcopy
from itertools import chain
import numpy as np
import sys

from distributions.distribution import TestDistribution
from util import bql, entropy, immutable

maxint = 2**63 - 2  # Largest integer random_integers can handle

# XXX: We need to make a CC engine with an explicit seed, here. Most of the
# entropy is generated in CC. See src/shell/src/main.py for an example of
# making a CC engine.

class BDB(TestDistribution):

    def __init__(self, sample=None, prngstate=None, numchains=None,
                 numiterations=None, bdb=None, reporter_bdb=None):
        if bdb is None:
            self.bdb = reporter_bdb or self._get_bdb()
            if reporter_bdb:
                reporter_bdb.truebdb = self._get_bdb()
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

    def _get_bdb(self):
        return bql.bdb_open()

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
        cmd = 'CREATE GENERATOR D_cc FOR D USING crosscat("c" numerical)'
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
        return np.array(self.bdb.exec_to_array(simcmd))

    def predictive_logpdf(self, value):
        "Return bdb-estimated logprob"
        probcmd = 'ESTIMATE PROBABILITY of c=%s BY D_cc' % value
        prob = self.bdb.exec_to_array(probcmd)
        assert len(prob) == 1
        return np.log(prob[0])


class BDBReporter(object):

    def execute(self, msg):
        print >> self.output, 'bdb.execute(%s)' % repr(msg)
        return self.truebdb.execute(msg)

    def sql_execute(self, msg):
        print >> self.output, 'bdb.sql_execute(%s)' % repr(msg)
        return self.truebdb.sql_execute(msg)

    def __getattribute__(self, name):
        if name in 'execute sql_execute truebdb output'.split():
            return object.__getattribute__(self, name)
        return object.__getattribute__(self.truebdb, name)


def make_reporter(*args, **kwargs):
    bdb = BDBReporter()
    bdb.output = kwargs.pop('output', sys.stdout)
    return BDB(*args, **dict(kwargs, reporter_bdb=bdb))
