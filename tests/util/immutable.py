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

If there are attributes of a class which you don't control and must be mutable,
you can register those by adding them to util.freeze.mutable_attributes, which
is a dictionary {<class>: <collection of names for mutable attributes>}.

If you are subclassing a class with mutable methods, you can specify those by
adding a class attribute __mutating_methods__, which should be a dictionary
{<method name>: <description of method>}.

Arguments passed to __init__ should be usually be immutable, and a rough check
of this is done by computing their hash, which is then used as the hash of the
instance. Once __init__ has been run, all subattributes of the instance are
recursively frozen by wrapping them in the Frozen class. If the arguments
should not be immutable (e.g. because you're wrapping a mutable class, as in
the Array class below), then you can provide a hash value via a
__compute_special_hash__ method. Don't override __compute_hash__, as it uses
name mangling.

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
import inspect
from itertools import chain
import numpy as np

from util import freeze
reload(freeze)
from util.meta_immutable import _MetaImmutable


class Immutable(object):

    __metaclass__ = _MetaImmutable
    __doc__ = __doc__  # Just use module docstring

    @staticmethod
    def unhashable_type_msg(exception):
        matches = [a for a in exception.args
                   if isinstance(a, basestring) and 'unhashable type' in a]
        if matches:
            assert len(matches) == 1, 'Should only by one error message'
            return matches.pop()
        return None

    def __compute_hash__(self):
        """Called by _MetaImmutable.__initialize_instance__ after performing
        initialization from __init__ method.

        Has to be defined in this context because of the way the __hash
        attribute name gets mangled.

        """
        if hasattr(self, '__compute_special_hash__'):
            self.__hash = self.__compute_special_hash__()
            return
        try:
            self.__hash = hash(self.__initial_values__)
        except TypeError, e:
            if self.unhashable_type_msg(e):
                # Iterate over the positional and keyword arguments to search
                # for the problematic argument
                args = chain(enumerate(self.__initial_values__[0]),
                             self.__initial_values__[1].items())
                for argidx, arg in args:
                    try:
                        hash(arg)
                    except TypeError, e:
                        message = self.unhashable_type_msg(e)
                        break
                msg = ('Immutable %s initialized with unhashable arguments: '
                       '"%s".  Complaint was about argument %i: %r.')
                raise TypeError(msg % (self.__class__.__name__, message,
                                       argidx, arg))
            else:  # Some other TypeError was raised.  Pass it on.
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
                                                  freeze.unequal)

    def __get_init_args__(self):
        posargnames = inspect.getargspec(self.__init__).args
        if posargnames.pop(0) != 'self':
            raise RuntimeError('Unexpected argument list')
        args = zip(posargnames, self.__initial_values__[0])
        # Return positional args + kwargs
        return args + self.__initial_values__[1].items()

    def __repr__(self):
        args = self.__get_init_args__()
        return '%s(%s)' % (self.__class__.__name__,
                           ', '.join('%s=%r' % (attrname, val)
                                     for attrname, val in args))


class Array(Immutable, np.ndarray):

    """A numpy array which is immutable (modulo warnings given above.)"""

    def __new__(self, input_array):
        # For an explanation of how this works, see the output of this command:
        # "help(np.doc.subclassing)".  Briefly, you have to convert a raw numpy
        # array in __new__ rather than create it in __init__.
        rv = np.asarray(input_array).copy().view(Array)
        # Override immutability to record type of initial array
        object.__setattr__(rv, '__initial_type__', type(input_array))
        return rv

    def __get_init_args__(self):
        return [('input_array', self.view(self.__initial_type__))]

    def __compute_special_hash__(self):
        # Arrays are unhashable, so need to pull its data in hashable format.
        # View as np.ndarray to avoid infinite loop when flattening, which
        # would otherwise create another 1-D Array, for which the hash would
        # need to be computed, which would result in another 1-D Array, etc.
        values = tuple(self.view(np.ndarray).flatten())
        return hash((values, self.shape, self.dtype))

    # Annotate the mutating methods for Immutable
    __method_groups = {'fill itemset put setfield': 'changing array values',
                       'partition reshape sort': 'rearranging array values',
                       'resize': 'changing dimensions',
                       'setflags': 'changing flags'}
    __mutating_methods__ = {}
    for attrnames, description in __method_groups.items():
        for attrname in attrnames.split():
            __mutating_methods__[attrname] = description

# Register these as types which do not need further freezing by
# freeze.deepfreeze.
freeze.frozen_types += (Immutable, Array)
