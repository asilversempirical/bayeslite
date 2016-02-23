"""Light protection against accidentally mutating an instance.

This is where you usually explain what the module is for, but let's be
frank: Immutability doesn't change a thing. :-)

Use this by inheriting from the Immutable class:

class MyClass(Immutable):
   ...etc...

When you do this, instances can be initialized as usual in their __init__
methods, but once __init__ is done all python special methods which would
typically mutate an instance (see protected_methods below) will raise an error.
This will prevent accidental mutation of an instance, though if you really need
to mutate something, e.g. for debugging during development, you can still do
it... E.g. object.__setattr__(myinstance, aname, avalue) will usually do what
myinstance.aname=avalue intends. If you do that in finished code you'll deserve
exactly whatever confusion you get, though, and you PROBABLY WON'T LIKE IT.

Arguments passed to __init__ should be immutable, and a rough check of this is
done by computing their hash, which is then used as the hash of the instance.
Once __init__ has been run, all subattributes of the instance are recursively
frozen by wrapping them in the Frozen class.

This class works by over-riding the instance construction procedure completely.
For background on how it works, see
https://blog.ionelmc.ro/2015/02/09/understanding-python-metaclasses/

In brief, given an Immutable subclass C, when C(*a, **kw) is called, the
__call__ method below is invoked. __call__ overwrites C's mutating special
methods with wrappers which check whether a flag has been set to indicate that
the class is initialized and no further mutation should be permitted. It then
obtains an instance c from C.__new__(*a, **kw). In c.__initial_values__, it
records the arguments (a, kw), and checks that these are hashable (and
therefore, hopefully, immutable.) Then it calls C.__init__(*a, **new) for
standard initialization. Finally, it sets a flag to indicate that
initialization is complete. Any special methods which would ordinarily be used
to mutate a class (e.g., __setattr__) are disabled by that flag.

Since instances are assumed to be immutable, pickling and unpickling are done
by simply storing __initial_values__, and calling __init__ on that when
restoring. Equality is also tested by comparing __initial_values__.

All this wrapping and unwrapping could be very slow, so it can be turned off by
setting __DEBUG__ to false.

"""

import collections
import copy
from frozendict import frozendict
from functools import partial
from numbers import Number
import types

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


def wraps(of, wf):
    """Assign useful metadata from function 'of' to wrapper wf."""
    # Logic taken from functools.wraps, but is a bit more robust
    attrs = ['__%s__' % name for name in 'name module doc'.split()]
    attrs.extend('im_' + n for n in 'class func self')
    for name in attrs:
        if hasattr(of, name):
            setattr(wf, name, getattr(of, name))


def freeze(o):
    if isinstance(o, frozen_types):
        return o
    if isinstance(o, collections.Mapping):
        return frozendict(o)
    if isinstance(o, collections.Set):
        return frozenset(o)
    if isinstance(o, collections.Sequence):
        return tuple(o)
    if callable(o):
        # Make sure the return value is frozen, too.
        def wrapper(*args, **kwargs):
            return freeze(o(*args, **kwargs))
        wraps(o, wrapper)
        o = wrapper
    return Frozen(o)

# Used to turn on logging in recursive deepfreeze
LOGFREEZE = False


def deepfreeze(o, memo=None):
    import logging
    if LOGFREEZE:
        # Warning level because I'm too lazy to configure logger
        logging.warning('Entering deepfreeze: ' + repr((o, memo)))
    memo = memo if memo is not None else {}
    f = partial(deepfreeze, memo=memo)
    _id = id(o)
    if _id in memo:
        return memo[_id]
    if isinstance(o, frozen_types):
        return o
    if isinstance(o, collections.Mapping):
        return freeze(dict((f(k), f(v)) for k, v in o.iteritems()))
    if isinstance(o, collections.Set):
        return freeze(set(f(e) for e in o))
    if isinstance(o, collections.Sequence):
        return freeze(list(f(e) for e in o))
    if hasattr(o, '__dict__'):
        # Freeze all attributes
        d = o.__dict__
        for k, v in d.iteritems():
            d[k] = deepfreeze(v)
    memo[_id] = freeze(o)
    if LOGFREEZE:
        logging.warning('Exiting deepfreeze: ' + repr((o, memo)))
    return memo[_id]


class UnusedObject:
    """Only use this as a placeholder which is not __eq__ to anything."""


class Frozen:

    """Immutable wrapper for arbitrary object."""

    def __init__(self, value):
        self._value = value

    def __getattribute__(self, name):
        if name in self.forbidden_methods:
            raise NotImplementedError('%s is forbidden on Frozen objects')
        v = (getattr(self._value, name) if name != '_value' else
             super(Frozen, self).__getattribute__('_value'))
        return freeze(v)

    def __hash__(self):
        return self._value.__hash__()

    def __eq__(self, other):
        return self._value == getattr(other, '_value', UnusedObject)

frozen_types += (Frozen,)


class MetaImmutable(type):

    # Just use the module docstring
    __doc__ = __doc__

    # Turn this off to disable the protections this class provides
    __DEBUG__ = __debug__

    def protect_getattr(klass, self):
        """Wrap self.__getattribute__, disabling special methods."""

        def __getattribute__(self, name):
            pname = protected_methods.get(name, False)
            if pname:
                raise TypeError('%s does not support %s' % (klass, pname))
            return freeze(super(klass, self).__getattribute__(name))

        wraps(self.__getattribute__, __getattribute__)
        # types.MethodType results in an instance-bound method
        self.__getattribute__ = types.MethodType(__getattribute__, self)

    def __call__(klass, *args, **kwargs):
        """This method is what gets called when you do C(*args, **kw) on a subclass C
        of Immutable.  It's by overriding this that we disable mutation methods.

        """
        # Create the instance
        self = klass.__new__(klass, *args, **kwargs)
        klass.__initialize_instance__(self, args, kwargs)
        return self

    def __initialize_instance__(klass, self, args, kwargs):
        """After performing initialization from __init__ method, make the
        instance immutable by disabling all special methods with mutation
        semantics.

        Record the __init__ arguments, and use their hash as the hash of
        the instance.  Finally, deepfreeze all attributes

        """
        # Do expected initialization
        self.__init__(*args, **kwargs)
        if klass.__DEBUG__:
            # Record initialization values, compute their hash
            self.__initial_values__ = deepfreeze((args, kwargs))
            try:
                self.__hash = hash(self.__initial_values__)
            except TypeError, e:
                # Verify error was due to unhashable type and report it if so
                strargs = (a for a in e.args if isinstance(a, basestring))
                if any('unhashable type' in a for a in strargs):
                    msg = 'Immutable %s initialized with unhashable arguments'
                    raise TypeError(msg % self.__class__.__name__)
                else:
                    raise e
            # Initialization is complete.  Disable mutating special methods
            klass.protect_getattr(self)
            # Freeze all instance attributes.  (Cannot freeze __dict__.)
            self.__dict__ = dict((k, deepfreeze(v))
                                 for k, v in self.__dict__.iteritems())
        else:
            self.__initial_values__ = (args, kwargs)


class Immutable(object):

    __metaclass__ = MetaImmutable
    __doc__ = __doc__  # Just use module docstring

    # __getinitargs__ is insufficient: attributes still pickled, and
    # initialize_instance won't happen...
    def __getstate__(self):
        """Get initial state for pickling...

        assumes the class has not been mutated since initialization. (So
        don't do that.)

        """
        return self.__initial_values__

    def __setstate__(self, state):
        args, kw = state
        # __initialize_instance__ isn't on the instance, it's on the metaclass.
        self.__class__.__initialize_instance__(self, args, kw)

    def __hash__(self):
        return self.__hash

    def __eq__(self, other):
        return self.__initial_values__ == getattr(other, '__initial_values__',
                                                  UnusedObject)

frozen_types += (Immutable,)
