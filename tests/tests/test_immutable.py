from cPickle import loads, dumps
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

    def __iadd__(self, o):
        self.a += o

    def __getitem__(self, k):
        return 9


def check_raised(f, E, m):
    try:
        f()
    except E, e:
        strargs = (a for a in e.args if isinstance(a, basestring))
        assert any(m in a for a in strargs), \
            ('Calling context should have raised %r with message '
             'containing %s, instead raised %s' % (E, m, e))

try:
    check_raised(lambda: 1 / 0., TypeError, 'anoetu')
except ZeroDivisionError:
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
    checked_operators = set()
    operators = '''mul:* div:/ floordiv:// mod:% pow:** lshift:<< rshift:>>
    and:& xor:^ or:| add:+ sub:-'''.split()
    operators = dict(o.split(':') for o in operators)
    for name, op in operators.iteritems():
        def f():
            exec ('i %s= 1' % op) in {'i': i}
        check_raised(f, TypeError, 'does not support in-place "%s"' % name)
        checked_operators.add(name)
    for op in 'truediv:1:truediv concat:[]:add repeat:1:mul'.split():
        op, val, ename = op.split(':')
        val = eval(val)

        def f():
            getattr(operator, '__i%s__' % op)(i, val)
        check_raised(f, TypeError, 'does not support in-place "%s"' % ename)
        checked_operators.add(op)
        all_operators = set(immutable.Immutable.ameths.split())
    assert checked_operators == all_operators, \
        'Checked all operators?  %s' % (all_operators - checked_operators)
    assert i.__getstate__() == i.__initial_values__
    assert loads(dumps(i)) == i, 'Does pickling invert?'
    check_raised(lambda: ImmutableTest({}, 2, 3), TypeError,
                 'initialized with unhashable arguments')
