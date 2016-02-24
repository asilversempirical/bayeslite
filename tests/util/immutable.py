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
obtains an instance c=C.__new__(*a, **kw). Then it calls C.__init__(c,*a,**new)
for standard initialization. In c.__initial_values__, it records immutable
references to the arguments (a, kw), and computes the hash of them to use as
the hash of c. Finally, it sets a flag to indicate that initialization is
complete, so that any special methods which would ordinarily be used to mutate
a class (e.g., __setattr__) are disabled before c is returned to the context in
which it was created.

Since instances are assumed to be immutable, pickling and unpickling are done
by simply storing __initial_values__, and calling __init__ on that when
restoring. Equality is also tested by comparing __initial_values__.

All this wrapping and unwrapping could be very slow, so it can be turned off by
setting __DEBUG__ to false. You should only do that for optimization in
production, though. Most of the time, you'll benefit from these defenses.

"""

from frozendict import frozendict
import logging

from util import freeze
from util.freeze import deepfreeze, protected_methods, unequal


def wraps(of, wf):
    """Assign useful metadata from function 'of' to wrapper wf."""
    # Logic taken from functools.wraps, but is a bit more robust
    attrs = ['__%s__' % name for name in 'name module doc'.split()]
    attrs.extend('im_' + n for n in 'class func self')
    for name in attrs:
        if hasattr(of, name):
            setattr(wf, name, getattr(of, name))


class _MetaImmutable(type):

    # Turn this off to disable the protections this class provides
    __DEBUG__ = __debug__

    # Turn this on to log calls to special method wrappers
    LOGWRAPS = False

    def wrap_method(klass, name, description):

        def notimplementederror(*args, **kw):
            raise NotImplementedError('class %s has no method %s' % (
                klass, name))

        method = getattr(klass, name, notimplementederror)
        if method == notimplementederror:
            method.__name__ = name
        elif hasattr(getattr(method, 'im_func', None),
                     '__immutable_wrapper__'):
            # Already a wrapper in a superclass; no need to make another
            return

        # Still in wrap_method
        def wrapped(self, *args, **kw):
            if klass.LOGWRAPS:
                logging.debug('%s.%s(*%s, **%s)' % (
                    self.__class__.__name__, name, args, kw))
            if hasattr(self, '__initialized__'):
                errmsg = '%s instance is Immutable and does not support %s' % (
                    self.__class__.__name__, description)
                raise TypeError(errmsg)
            return method(self, *args, **kw)

        # Still in wrap_method
        wraps(method, wrapped)  # Transfer method metadata to wrapper
        setattr(klass, name, wrapped)
        # Signal that this is already a wrapper which checks for immutability.
        # Can't put this on the method itself, not allowed.
        setattr(getattr(klass, name).im_func, '__immutable_wrapper__', True)

    def __init__(klass, name, bases, attrs, **kwargs):
        """Called during initialization of client class, not its instances.

        We use this to set the wrappers for the mutating special
        methods.

        """
        # Provide wrappers for mutating special methods
        for name, description in protected_methods.iteritems():
            klass.wrap_method(name, description)

    def __call__(klass, *args, **kwargs):
        """This method is called when you do C(*args, **kw) on a subclass C of
        Immutable. It's by overriding this that we disable mutation methods.

        """
        # Create the instance
        self = klass.__new__(klass, *args, **kwargs)
        klass.__initialize_instance__(self, args, kwargs)
        return self

    def __initialize_instance__(klass, self, args, kwargs):
        # Do expected initialization
        self.__init__(*args, **kwargs)
        if klass.__DEBUG__:
            # Record the (frozen) initial values
            self.__initial_values__ = deepfreeze((args, kwargs))
            self.__compute_hash__()
            # Turn on immutability
            klass.__freeze_all_attributes__(self)
            self.__initialized__ = True
        else:
            # Record the initial values (fast but unsafe: potentially mutable.)
            self.__initial_values__ = (args, kwargs)

    def __freeze_all_attributes__(klass, self):
        # (Cannot freeze __dict__.)
        self.__dict__ = dict((k, deepfreeze(v))
                             for k, v in self.__dict__.iteritems())


class Immutable(object):

    __metaclass__ = _MetaImmutable
    __doc__ = __doc__  # Just use module docstring

    def __compute_hash__(self):
        """Called by _MetaImmutable.__initialize_instance__ after performing
        initialization from __init__ method.

        Has to be defined in this context because of the way the __hash
        attribute name gets mangled.

        """
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

    def __getattribute__(self, name):
        rv = super(Immutable, self).__getattribute__(name)
        return frozendict(rv) if name == '__dict__' else rv

    # For pickling machinery, __getinitargs__ is insufficient: if you supply it
    # all attributes are still pickled, and initialize_instance won't happen...
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
                                                  unequal)

# Register this as a type which does not need further freezing by
# freeze.deepfreeze.
freeze.frozen_types += (Immutable,)
