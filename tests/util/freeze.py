"""Utilities for wrapping python objects with immutable proxies.

The main entry point is deepfreeze, which will walk down an object and
wrap everything in it with a proxy which does not allow mutating special
methods. Do not use _freeze or _Frozen directly. Their return values can
contain mutable attributes.

"""

import collections
import copy
from frozendict import frozendict
from functools import partial
from itertools import chain
import logging
from numbers import Number

from scipy.stats._distn_infrastructure import rv_frozen
from numpy import ndarray

# A mapping of classes to attributes which should not be made immutable,
# because their implementations can't handle it.
mutable_attributes = {rv_frozen: set(['kwds'])}

# A mapping of method names which which should be protected from accidental
# access in Immutable instances because their default semantics imply mutation.
# Maps to english names for them (for error messages.)
protected_methods = {'__set__': 'property assignment',
                     '__delete__': 'property deletion'}
# Add attribute, item and slice mutation operations to the list
mutetargets = {'attr': 'attribute', 'item': 'item', 'slice': 'slice'}
for obj, name in mutetargets.iteritems():
    for verb, vn in {'set': 'assignment', 'del': 'deletion'}.iteritems():
        protected_methods['__%s%s__' % (verb, obj)] = '%s %s' % (name, vn)
        # Add in-place arithmetic operations to the protected list
ameths = '''mul div truediv floordiv mod pow lshift rshift and xor or
add concat repeat sub'''
for method in ameths.split():
    protected_methods['__i%s__' % method] = 'in-place "%s"' % method

# List of types which are immutable to start with, and therefore don't need to
# be wrapped.
frozen_types = set(v for v, d in copy._deepcopy_dispatch.iteritems()
                   if d == copy._deepcopy_atomic)
# tuple, frozendict and FrozenList can contain mutable elements, shouldn't be
# assumed immutable. Theoretically, frozenset could contain something which is
# hashable but mutable, too, but if people are going to break the semantics
# like that there is not much we can do.
frozen_types.update((Number, basestring, buffer, memoryview, slice,
                     frozenset))
frozen_types = tuple(frozen_types)  # Needs to be tuple for isinstance usage

# Used to turn on logging in recursive deepfreeze
LOGFREEZE = False
logging.basicConfig(filename='/tmp/freeze.log')
logging.getLogger().setLevel(logging.DEBUG)  # to display in REPL


class UnEqual:

    """Only use this as a placeholder which is not __eq__ to anything else."""

    def __eq__(klass, other):
        return False

unequal = UnEqual()


def get_mutable_attributes(o):
    return set(chain(*(m for k, m in mutable_attributes.items()
                       if isinstance(o, k))))


class _Frozen(object):

    """Immutable wrapper for arbitrary object. Don't use this directly, use
    deepfreeze. This only makes the top level references immutable.

    """

    def __init__(self, value):
        self._value = value

    def __getattribute__(self, name):
        v = super(_Frozen, self).__getattribute__('_value')
        if name == '_value':
            return v
        if LOGFREEZE:
            logging.debug('in __getattribute__: %r, %s' % (self, name))
        if name in protected_methods:
            raise NotImplementedError('%s is forbidden on Frozen objects')
        rv = getattr(self._value, name)
        if name in get_mutable_attributes(v):
            return rv
        return _freeze(rv)

    def __hash__(self):
        return hash(self._value)

    def __eq__(self, other):
        return self._value == getattr(other, '_value', unequal)

    def __call__(self, *args, **kw):
        return _freeze(self._value(*args, **kw))

frozen_types += (_Frozen,)


def _freeze(o):
    """Freeze the top level references of o if it is a collection. Don't use this
    directly, use deepfreeze."""
    if LOGFREEZE:
        logging.debug('entering _freeze: ' + repr(o))
    if isinstance(o, frozen_types):
        return o
    if isinstance(o, collections.Mapping):
        return frozendict(o)
    if isinstance(o, collections.Set):
        return frozenset(o)
    if isinstance(o, collections.Sequence):
        return tuple(o)
    if isinstance(o, ndarray):
        rv = o.copy()
        rv.setflags(write=False)
        return rv
    if LOGFREEZE:
        logging.debug('exiting _freeze: ' + repr(o))
    return _Frozen(o)


def deepfreeze(o, memo=None):
    """Walk into all references reachable from o, and wrap those references in
    immutable proxies.

    The return value is something which ought to be hard to accidentally
    mutate, though you'll be able to if you insist.

    """
    if LOGFREEZE:
        logging.debug('Entering deepfreeze: ' + repr((o, memo)))
    # memo is used to break cycles in the reference graph.
    memo = memo if memo is not None else {}
    f = partial(deepfreeze, memo=memo)
    _id = id(o)
    if _id in memo:
        return memo[_id]
    if isinstance(o, frozen_types):
        return o
    if isinstance(o, collections.Mapping):
        return _freeze(dict((f(k), f(v)) for k, v in o.iteritems()))
    if isinstance(o, collections.Set):
        return _freeze(set(f(e) for e in o))
    if isinstance(o, collections.Sequence):
        return _freeze(list(f(e) for e in o))
    if hasattr(o, '__dict__'):
        mutables = get_mutable_attributes(o)
        # Freeze all immutable attributes. Can't freeze the dict, but it will
        # be returned frozen by _Frozen.__getattribute__
        for k, v in o.__dict__.iteritems():
            if k not in mutables:
                o.__dict__[k] = f(v)
    memo[_id] = _freeze(o)
    if LOGFREEZE:
        logging.debug('Exiting deepfreeze: ' + repr((o, memo)))
    return memo[_id]
