from bdbcontrib import cursor_to_df
from util import bdb

class BayesDB(bdb.BayesDB):

    def exec_to_df(self, *args, **kw):
        return cursor_to_df(self.execute(*args, **kw))

    def sql_exec_to_df(self, *args, **kw):
        return cursor_to_df(self.sql_execute(*args, **kw))

def bdb_open(*args, **kw):
    rv = bdb.bayesdb_open(*args, **dict(kw, bdbclass=BayesDB))
    return rv
