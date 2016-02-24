from cPickle import loads, dumps
from frozendict import frozendict
from functools import partial
import operator

from util import immutable


class ImmutableTest(immutable.Immutable):

    def __init__(self, a, b, c, k=20):
        self.a, self.b, self.c, self.k = a, b, c, k

    def mutating_method(self, d):
        self.d = d

    def nonmutating_method(self, d):
        return self.a + self.k

    def __getitem__(self, item):
        return 9

    def geta(self):
        return self.a

    def seta(self, v):
        self.a = v

    def dela(self):
        del self.a

    ab = property(geta, seta, dela, 'the a property')


def check_raised(f, E, m):
    try:
        f()
    except E, e:
        strargs = (a for a in e.args if isinstance(a, basestring))
        assert any(m in a for a in strargs), \
            ('Calling context should have raised %r with message '
             'containing "%s", instead raised "%s"' % (E, m, e))
    else:
        raise RuntimeError('Failed to raise %s(%%%s%%)' % (E.__name__, m))

try:
    check_raised(lambda: 1 / 0., ZeroDivisionError, 'anoetu')
except AssertionError:
    """This is the correct path."""
else:
    raise RuntimeError('Positive control failed')


def test_immutable(C=ImmutableTest):
    i = C(1, 2, 3)
    i.nonmutating_method(6)  # Should run without error
    check_raised(partial(i.mutating_method, 5),
                 TypeError, 'does not support attribute assignment')
    check_raised(lambda: delattr(i, 'a'), TypeError,
                 'does not support attribute deletion')

    def setitem():
        i[5] = 6
    check_raised(setitem, TypeError, 'does not support item assignment')

    def delitem():
        del i[5]
    check_raised(delitem, TypeError, 'does not support item deletion')

    def setslice():
        i[5:6] = [1, 2]
    check_raised(setslice, TypeError, 'does not support slice assignment')

    def delslice():
        del i[5:6]
    check_raised(delslice, TypeError, 'does not support slice deletion')

    def setprop():
        i.ab = 7
    check_raised(setprop, TypeError, 'does not support attribute assignment')

    def delprop():
        del i.ab
    check_raised(delprop, TypeError, 'does not support attribute deletion')
    checked_operators = set(['__delitem__', '__setitem__', '__setattr__',
                             '__delattr__', '__setslice__', '__delslice__',
                             '__set__', '__delete__'])
    operators = '''mul:* div:/ floordiv:// mod:% pow:** lshift:<< rshift:>>
    and:& xor:^ or:| add:+ sub:-'''.split()
    operators = dict(o.split(':') for o in operators)
    for name, op in operators.iteritems():
        def f():
            exec ('i %s= 1' % op) in {'i': i}
        check_raised(f, TypeError, 'does not support in-place "%s"' % name)
        checked_operators.add('__i%s__' % name)
    for op in 'truediv:1:truediv concat:[]:add repeat:1:mul'.split():
        op, val, ename = op.split(':')
        val = eval(val)

        def f():
            getattr(operator, '__i%s__' % op)(i, val)
        check_raised(f, TypeError, 'does not support in-place "%s"' % ename)
        checked_operators.add('__i%s__' % op)
    all_operators = set(immutable.protected_methods)
    assert checked_operators == all_operators, \
        'Checked all operators?  %s' % (all_operators - checked_operators)
    assert i.__getstate__() == i.__initial_values__
    assert loads(dumps(i)) == i, 'Does pickling invert?'
    hash(ImmutableTest({}, 2, 3))
    assert isinstance(i.__dict__, frozendict), \
        'Is __dict__ being frozen?'


def test_deepfreeze():
    from frozendict import frozendict
    rv = immutable.deepfreeze(((1,), {}))
    assert rv == ((1,), frozendict({})), \
        ('Got %r, expected ((1,), frozendict())' % (rv,))


class DEBUGTest(ImmutableTest):

    __DEBUG__ = False


def test_debug():
    i = DEBUGTest(1, 2, 3, k=5)
    i.a = 1  # Check that __DEBUG__==False disables protections

def main():
    test_immutable()
    test_deepfreeze()
    test_debug()
