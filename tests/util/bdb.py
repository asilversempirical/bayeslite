"A pickleable and copyable bdb"

import tempfile

from bayeslite import bayesdb
from bayeslite.metamodel import bayesdb_register_metamodel as register
from bayeslite.metamodel import builtin_metamodels as models

class BayesDB(bayesdb.BayesDB):

    "A pickleable and copyable bdb"

    def __init__(self, *args, **kwargs):
        # Make sure we have a handy way to copy a database, by putting it on
        # disk. It would be possible to dump out in-memory DBs to disk; see
        # http://apidoc.apsw.googlecode.com/hg/example.html for details.
        if kwargs.get('pathname', None) is None:
            kwargs['pathname'] = tempfile.NamedTemporaryFile().name
            super(BayesDB, self).__init__(*args, **kwargs)
            self.version = kwargs['version']
            self.compatible = kwargs['compatible']

    def __getstate__(self):
        self.close()
        state = self.__dict__.copy()
        # Remove db-related material from pickle state (can't pickle.)
        for badkey in '_sqlite3 _empty_cursor'.split():
            del state[badkey]
        assert state['cache'] is None
        # Record the *names* of the metamodels. Keep their state out of it.
        # registering and deregistering a metamodel, at this stage,
        # registration ONLY hits the DB. It doesn't have any impact on
        # metamodel internal state.
        state['metamodel_names'] = set(state.pop('metamodels').keys())
        # Store the db in the state
        state['db'] = open(self.pathname).read()
        # Reconnect to the database
        self.connect()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Take a disk-based copy of the db.
        dbfile = tempfile.NamedTemporaryFile('w')
        dbfile.write(self.db)
        dbfile.close()
        self.connect()
        # Register the new bdb with all the right metamodels
        self.metamodels = {}
        for model in models:
            if model.name() in self.metamodel_names:
                register(self, model)
        del self.metamodels, self.db

def bayesdb_open(*args, **kw):
    bdbclass = kw.get('bdbclass', BayesDB)
    rv = bayesdb.bayesdb_open(*args, **dict(kw, bdbclass=bdbclass))
    return rv
