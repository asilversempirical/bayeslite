from bdbcontrib import cursor_to_df
from util import bdb

class BayesDB(bdb.BayesDB):

    def exec_to_df(self, *args, **kw):
        return cursor_to_df(self.execute(*args, **kw))

    def sql_exec_to_df(self, *args, **kw):
        return cursor_to_df(self.sql_execute(*args, **kw))

    def exec_to_array(self, cmd, column_name=None):
        rv = self.exec_to_df(cmd)
        if column_name is None:
            column_name = rv.columns[0]
        return getattr(rv, column_name)

def bdb_open(*args, **kw):
    rv = bdb.bayesdb_open(*args, **dict(kw, bdbclass=BayesDB))
    return rv
