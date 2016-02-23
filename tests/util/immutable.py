"""Light protection against accidentally mutating an instance.

This is where you usually explain what the module is for, but let's be
frank: Immutability doesn't change a thing. :-)

Use this by inheriting from the Immutable class:

class MyClass:
   __metaclass__ = Immutable

When you do this, instances can be initialized as usual in their __init__
methods, but once __init__ is done all python special methods which would
typically mutate an instance (see protected_methods below) will raise an error.
This will prevent accidental mutation of an instance, though if you really need
to mutate something, e.g. for debugging during development you can still do
it... E.g. object.__setattr__(myinstance, aname, avalue) will usually do what
myinstance.aname=avalue intends. If you do that in finished code you'll deserve
exactly whatever confusion you get, though, and you probably won't like it
eventually.

It also doesn't stop you from assigning mutable attributes during instance
contruction, and then changing those attributes. Just... don't do that. It does
at least check for you that none of the arguments you passed to the class are
mutable, by computing their hash.

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

"""

from functools import wraps, partial


class MetaImmutable(type):

    # Just use the module docstring
    __doc__ = __doc__

    # A mapping of method names which which should be protected from accidental
    # access, along with english names for them (for error messages.) For use
    # with protected_special_method.
    protected_methods = {'set': 'attribute property assignment'}
    # Add attribute, item and slice mutation operations to the list
    mutetargets = {'attr': 'attribute', 'item': 'item', 'slice': 'slice'}
    for obj, name in mutetargets.iteritems():
        for verb, vn in {'set': 'assignment', 'del': 'deletion'}.iteritems():
            protected_methods[verb + obj] = '%s %s' % (name, vn)
    # Add in-place arithmetic operations to the protected list
    ameths = '''mul div truediv floordiv mod pow lshift rshift and xor or
    add concat repeat sub'''
    for method in ameths.split():
        protected_methods['i%s' % method] = 'in-place "%s"' % method

    @staticmethod
    def protected_special_method(klass, mname, engname):
        """Wrap the method mname of klass so that it will fail when called on a
        klass instance which has already been initialized, with error message
        including engname as the English description of the method
        semantics."""

        def method(self, *args, **kwargs):
            if not getattr(self, '__initialized__', False):
                super_method = getattr(super(klass, self), mname)
                return super_method(*args, **kwargs)
            else:
                raise TypeError('%s does not support %s' % (klass, engname))

        if hasattr(klass, mname):
            meth = getattr(klass, mname)
            for name in 'module doc name'.split():
                name = '__%s__' % name
                if hasattr(meth, name):
                    setattr(method, name, getattr(meth, name))
        return method

    def __call__(klass, *args, **kwargs):
        # Protect all mutating special methods
        protect = partial(klass.protected_special_method, klass)
        for mname, engname in klass.protected_methods.iteritems():
            mname = '__%s__' % mname
            method = protect(mname, engname)
            setattr(klass, mname, method)
        # Create the class instance
        self = klass.__new__(klass)
        # Record the initial values and check they're hashable
        klass.__initial_values__ = (args, tuple(kwargs.iteritems()))
        try:
            self.__hash = hash(self.__initial_values__)
        except TypeError, e:
            msg = '%s initialized with unhashable arguments'
            strargs = (a for a in e.args if isinstance(a, basestring))
            if any('unhashable type' in a for a in strargs):
                raise TypeError(msg % self.__class__.__name__)
            else:
                raise e
        # Do the expected initialization
        klass.__init__(self, *args, **kwargs)
        # Record that this instance is now initialized, and therefore
        # immutable.
        self.__initialized__ = True
        return self


class NonObject:
    """Something nothing else should be equal to."""


class Immutable(object):

    __metaclass__ = MetaImmutable

    def __getstate__(self):
        """Get initial state for pickling...

        assumes the class has not been mutated since initialization. (So
        don't do that.)

        """
        return self.__initial_values__

    def __setstate__(self, state):
        args, kwargs = self.__initial_values__
        self.__init__(*args, **dict(kwargs))

    def __hash__(self):
        return self.__hash

    def __eq__(self, other):
        return self.__initial_values__ == getattr(other, '__initial_values__',
                                                  NonObject)
