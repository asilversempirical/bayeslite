# https://blog.ionelmc.ro/2015/02/09/understanding-python-metaclasses/

"""Light protection against accidentally mutating an instance.

This is where you usually explain what the module is for, but let's be
frank: Immutability doesn't change a thing. :-)

Use this by setting the metaclass of a class you want to Immutable:

class MyClass:
   __metaclass__ = Immutable

When you do this, instances can be initialized as usual in their __init__
methods, but once __init__ is done all python special methods which would
typically mutate an instance (see protected_methods below) will raise an error.
This will prevent accidental mutation of an instance, though if you really need
to mutate something, e.g. for debugging during development you can still do
it... E.g. object.__setattr__(myinstance, aname, avalue) will usually do what
myinstance.aname=avalue intends. If you do that in finished code you'll deserve
exactly whatever confusion you get, though, and you probably won't like it.

It also doesn't stop you from assigning mutable attributes during instance
contruction, and then changing those attributes. Just... don't do that. It does
at least check for you that none of the arguments you passed to the class are
mutable, by computing their hash.

This class works by over-riding the instance construction procedure completely
-- currently, if your class C defines __new__ and has Immutable as
__metaclass__, __new__ will not be called (but this would be easy to correct.)

When C(*a, **kw) is called, the __call__ method below is invoked. It checks
whether the instance has already been instantiated, and if so, passes its
arguments on to C.__call__. If not, it assumes a new instance is being created
and passes its arguments on to C.__init__. On return from C.__init__, it sets a
flag to indicate that the class is now immutable. Any special methods which
would ordinarily be used to mutate a class (e.g., __setattr__) are disabled by
that flag.

Since instances are assumed to be immutable, pickling and  unpickling are

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
        """Wrap the method mname of klass so that it will fail when called on a klass
        instance which has already been initialized, with error message
        including engname as the English description of the method semantics.

        """

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
        self = klass.__new__(klass)
        # Protect all mutating special methods
        protect = partial(klass.protected_special_method, klass)
        for mname, engname in klass.protected_methods.iteritems():
            mname = '__%s__' % mname
            method = protect(mname, engname)
            setattr(klass, mname, method)
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
    "Something nothing else should be equal to"

class Immutable(object):

    __metaclass__ = MetaImmutable

    def __getstate__(self):
        """Get initial state for pickling... assumes the class has not been mutated
        since initialization. (So don't do that.)

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
