"""Utilities for wrapping python objects with immutable proxies.

This is mainly for use by the util.immutable module, and you should read its
documentation first. The main entry point here is deepfreeze, which will walk
down an object and wrap everything in it with a proxy which does not allow
mutating special methods. Do not use _freeze or _Frozen directly. Their return
values can contain mutable attributes.

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
from bayeslite.metamodels.crosscat_theta_validator import Validator

# A mapping of classes to attributes which should not be made immutable,
# because their implementations can't handle it.
mutable_attributes = {rv_frozen: set(['kwds']), Validator: set(['schema'])}

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
LOGFREEZE = True
logging.basicConfig(filename='/tmp/freeze.log')  # To log to file
logging.getLogger().setLevel(logging.DEBUG)  # to enable logging


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
        self.__value = value

    def __getattribute__(self, name):
        # Manual mangling
        v = super(_Frozen, self).__getattribute__('_Frozen__value')
        if name == '_Frozen__value':
            return v
        if LOGFREEZE:
            logging.debug('in __getattribute__: %r, %s' % (self, name))
        if name in protected_methods:
            raise NotImplementedError('%s is forbidden on Frozen objects')
        rv = getattr(v, name)
        if name in get_mutable_attributes(v):
            return rv
        return deepfreeze(rv)

    def __hash__(self):
        return hash(self.__value)

    def __eq__(self, other):
        return self.__value == getattr(other, '_Frozen__value', unequal)

    def __call__(self, *args, **kw):
        return deepfreeze(self.__value(*args, **kw))

    def __repr__(self):
        # Get __value directly to prevent infinite recursion when logging in
        # __getattribute__
        value = object.__getattribute__(self, '_Frozen__value')
        return '<Frozen %r>' % value

frozen_types += (_Frozen,)

# This is replaced by the immutable module, breaking the circular depedency.
Array = None


def deepfreeze(o, memo=None, seen=None):
    """Walk into all references reachable from o, and wrap those references in
    immutable proxies.

    The return value is something which ought to be hard to accidentally
    mutate, though you'll be able to if you insist.

    """
    if isinstance(o, frozen_types):
        return o
    if LOGFREEZE:
        logging.debug('Entering deepfreeze: ' + repr((o, memo)))
    # memo is used to break cycles in the reference graph.
    memo = memo if memo is not None else {}
    seen = seen if seen is not None else set()
    f = partial(deepfreeze, memo=memo, seen=seen)
    _id = id(o)
    assert _id not in seen, 'Infinite loop!'
    seen.add(_id)
    if _id in memo:
        return memo[_id]
    if isinstance(o, ndarray):
        frozen = Array(o)
    elif isinstance(o, collections.Mapping):
        frozen = frozendict((f(k), f(v)) for k, v in o.iteritems())
    elif isinstance(o, collections.Set):
        frozen = frozenset(f(e) for e in o)
    elif isinstance(o, collections.Sequence):
        frozen = tuple(f(e) for e in o)
    else:
        frozen = _Frozen(o)
    memo[_id] = frozen
    return frozen
    if LOGFREEZE:
        logging.debug('Exiting deepfreeze: ' + repr((o, memo)))
    return memo[_id]
