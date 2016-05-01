def safezip(*tozip):
    "Like zip, but throws if any arguments have different lengths"
    iterators = map(iter, tozip)
    rv = zip(*iterators)
    for lidx, iterator in enumerate(iterators):
        try:
            iterator.next()
        except StopIteration:
            pass
        else:
            raise ValueError('Argument %i has extra elements' % lidx)
    return rv
