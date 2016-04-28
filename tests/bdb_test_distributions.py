"""A wrapper around a bdb instance using the GPM interface.

Although the system wrapped up in a BayesDB instance is extremely protean, we
can make all the public methods here referentially transparent by doing all the
training during initialization from a fresh bdb.

When a `BayesDBTestDistribution` is initialized, a fresh bdb is created, seeded
according to prngstate , and populated with the sample, and trained on it on
numchains chains for numiterations iterations.

"""

import numpy as np
import random
import re
import struct

from bayeslite.bayesdb import bayesdb_open
from bayeslite import weakprng


class BayesDBTestDistribution:

    def __init__(self, sample, prngstate, numchains, numiterations):
        self.bdb = bayesdb_open()
        self.seed(prngstate)
        self.sample = sample
        self._populate_model()
        self._create_chains(numchains, prngstate)
        self._analyze(numiterations, prngstate)

    def seed(self, prngstate):
        "Set the seeds for the bdb's RNGs"
        seed = [prngstate.randint(0, 2**32 - 1) for _ in xrange(8)]
        self.bdb._prng = weakprng.weakprng(struct.pack('IIIIIIII', *seed))
        self.bdb._py_prng = random.Random(np.prod(seed))
        self.bdb._np_prng = np.random.RandomState(seed)

    def forbidden(self, message):
        def forbidden(message):
            raise RuntimeError(message)
        return forbidden

    def _populate_model(self):
        "Put the given data into a table so that the bdb can operate on it"
        self.bdb.sql_execute('CREATE TABLE D (c float)')
        insertcmd = 'INSERT INTO D (c) VALUES ' + \
                    ', '.join(len(self.sample) * ['(?)'])
        self.bdb.sql_execute(insertcmd, self.sample)
        self._populate_model = self.forbidden('Already populated')

    def _create_chains(self, numchains, prngstate):
        "Create the model states the MCMC chains will iterate on."
        self.seed(prngstate)
        cmd = 'CREATE GENERATOR D_cc FOR D USING crosscat("c" numerical)'
        self.bdb.execute(cmd)
        # XXX: Can't parameterize the initialization command -- leads to BQL
        # syntax error. So rely on interpolation of '%i' to validate the input.
        self.bdb.execute('INITIALIZE %i MODELS for D_cc' % numchains)
        self._create_chains = self.forbidden('Already created')

    def _analyze(self, numiterations, prngstate=None):
        "Run through numiterations MCMC steps."
        self.seed(prngstate)
        # XXX: Can't parameterize the analyze command -- leads to BQL
        # syntax error. So rely on interpolation of '%i' to validate the input.
        self.bdb.execute('ANALYZE D_cc FOR %i ITERATIONS WAIT' % numiterations)
        self._analyze = self.forbidden('Already analyzed')

    def exec_to_array(self, cmd, parameters=None):
        "Return the single-column results from executing `cmd`."
        return np.array(list(self.bdb.execute(cmd, parameters))).T[0]

    def simulate(self, samplesize, prngstate=None):
        "Draw samplesize variates from the estimated posterior distribution"
        self.seed(prngstate)
        # XXX: Can't parameterize the analyze command -- leads to BQL
        # type error. So rely on interpolation of '%i' to validate the input.
        simcmd = 'SIMULATE c from D_cc LIMIT %i' % samplesize
        return np.array(self.exec_to_array(simcmd))

    def predictive_logpdf(self, value):
        "Return bdb-estimated logprob"
        # XXX: Can't parameterize the estimate command -- leads to BQL
        # type error. So rely on interpolation of '%f' to validate the input.
        probcmd = 'ESTIMATE PROBABILITY of c=%f BY D_cc' % value
        prob = self.exec_to_array(probcmd)
        assert len(prob) == 1
        return np.log(prob[0])
