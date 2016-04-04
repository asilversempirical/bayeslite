'''Parser for tokenized crosscat generator expressions.'''

import collections
import numbers
import types

from bayeslite.exception import BQLError, BQLParseError
from bayeslite.util import casefold

# guess is bool. subsample is False or an int. columns is a list of pairs
# (column name, type). dep_constraints is a list of (column names, dep), where
# column names is a list of column names and dep is a bool indicating whether
# they're dependent or independent.
GeneratorSchema = collections.namedtuple(
    'GeneratorSchema',
    ['guess', 'columns', 'dep_constraints', 'subsample', 'subsample_seed'])

def parse(schema, subsample_default):
    '''Parses a generator schema as passed to CrosscatMetamodel.

    schema is a tokenized expression of the form [['GUESS', ['*']], ['x',
    'NUMERICAL'], ...] that is passed to CrosscatMetamodel.create_generator and
    represents the argument to "crosscat" in CREATE GENERATOR ... FOR ... USING
    crosscat(...).

    Returns a GeneratorSchema.

    See test_crosscat_generator_schema.py for examples.
    '''

    guess = False
    subsample = subsample_default
    subsample_seed = None
    columns = []
    dep_constraints = []
    for directive in schema:

        if directive == []:
            # Skip extra commas so you can write
            #
            #    CREATE GENERATOR t_cc FOR t USING crosscat(
            #        x,
            #        y,
            #        z,
            #    )
            continue

        if (not isinstance(directive, list) or len(directive) != 2 or
                not isinstance(directive[0], basestring)):
            raise BQLError(
                None,
                'Invalid crosscat column model directive: %r' % (directive,))

        op = casefold(directive[0])
        if op == 'guess' and directive[1] == ['*']:
            guess = True
        elif op == 'subsample':
            subsample, subsample_seed = _parse_subsample_clause(*directive[1])
        elif op == 'dependent':
            constraint = (_parse_dependent_clause(directive[1]), True)
            dep_constraints.append(constraint)
        elif op == 'independent':
            constraint = (_parse_dependent_clause(directive[1]), False)
            dep_constraints.append(constraint)
        elif op != 'guess' and casefold(directive[1]) != 'guess':
            columns.append((directive[0], directive[1]))
        else:
            raise BQLError(
                None, 'Invalid crosscat column model: %r' % (directive),)
    return GeneratorSchema(guess=guess, columns=columns,
        dep_constraints=dep_constraints,
        subsample=subsample, subsample_seed=subsample_seed)


# Use argument destructuring to parse directive
def _parse_subsample_clause(off_or_size, comma=None, seed=None, *rest):
    args = (off_or_size, comma, seed) + rest
    if rest != ():  # Implies more than 2 arguments were given
        raise BQLError(None, ('subsample takes 1 or 2 arguments: '
                              'subsample size or "off" and (optionally) random'
                              ' seed: %r' % args))
    if (comma is not None) and (comma != ','):
        raise BQLParseError('Expected a comma, or nothing.')
    if casefold(off_or_size) == 'off':
        if seed is not None:
            raise BQLError(None, ('Subsample seed only makes sense when '
                                  'subsampling is enabled: %r' % args))
        return False, None
    if not (isinstance(off_or_size, numbers.Integral) and
            isinstance(seed, (types.NoneType, numbers.Integral))):
        raise BQLError(None, ('Either pass "off" to subsample, or one or two '
                              'integers'))
    return off_or_size, seed


def _parse_dependent_clause(args):
    i = 0
    dep_columns = []
    while i < len(args):
        dep_columns.append(args[i])
        if i + 1 < len(args) and args[i + 1] != ',':
            raise BQLError(None, 'Invalid dependent columns: %r' % (args,))
        i += 2
    return dep_columns
