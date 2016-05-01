import numpy as np

def sample_dp(alpha, maxlen, prngstate):
    """Draw a multinomial from the Dirichlet Process with concentration parameter
    `alpha`, truncated to at most `maxlen` events.

    """
    remaining = 1.
    rv = []
    while (remaining > 1e-20) and (len(rv) < maxlen):
        wi = prngstate.dirichlet([1, alpha])[0]
        length = wi*remaining
        rv.append(length)
        remaining -= length
    total = sum(rv)
    assert (min(rv) > 0) and (total <= 1)
    return np.array(rv) / total
