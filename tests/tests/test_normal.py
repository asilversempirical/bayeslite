"""Tests of normal distribution and inverse-gamma prior"""

import numpy as np
from numpy import log, abs, exp
import scipy

from distributions.normal import Normal, NormalInvGammaPrior
from util.entropy import seeded, random_state
from util import statistic_threshold as tst
from util.statistics import kullback_leibler, kernel_two_sample_statistic


def normal_prior_check_statistic(prngstate):
    """Test statistic which sanity checks NormalInvGammaPrior by verifying that
    the actual mean & variance from which a sample is drawn lie in the likely
    region of the posterior given the sample."""
    mu, var, samplesize = 5, 2, 100
    n = Normal(mu, var)
    sample = n.simulate(samplesize, prngstate=prngstate)
    prior = NormalInvGammaPrior(mu, 1. / var, 1, 1)
    posterior = prior.posterior(sample)
    return min(posterior.sigma.sf(var),
               posterior.mudist(var, prngstate).n.sf(mu))

ntest_statistic = seeded(normal_prior_check_statistic, 17)
# tst.compute_sufficiently_stringent_threshold(ntest_statistic, 6, 1e-15)
# => (0.0013068798846691543, 9.628012077249724e-16, 42478)
ntest_thresh = 1.30688e-3


def test_normal_prior():
    tst.check_generator(ntest_statistic, 6, ntest_thresh, 1e-15)


def kullback_leibler_normal(number_doublings):
    """Get the data for a visualization showing the convergence of the
    NormalInvGammaPrior posteriors as the sample size converges."""
    kl_divs, mmds, posteriors = [], [], []
    prngstate = random_state(17)
    mu, var = 0, 1
    n = Normal(mu, var)
    sample = list(n.simulate(1, prngstate=prngstate))
    samplesizes = 2**scipy.arange(0, number_doublings, 1)
    lowbound, highbound = -5, 5
    x = np.linspace(lowbound, highbound, 100)
    X, Y = np.meshgrid(x, log(samplesizes) / log(2))
    Z = scipy.zeros(X.shape)
    for sidx, samplesize in enumerate(samplesizes):
        assert len(sample) == samplesize
        prior = NormalInvGammaPrior(mu, 1. / var, 1, 1)
        posteriors.append(prior.posterior(sample))
        posterior = posteriors[-1].predictive_logpdf
        kl_divs.append(kullback_leibler(sample, posterior, n.logpdf)[0])
        post_sample = posteriors[-1].simulate(samplesize, prngstate=prngstate)
        px_sample = [s.simulate(prngstate=prngstate) for s in post_sample]
        mmd = kernel_two_sample_statistic(
            scipy.array(px_sample), scipy.array(sample, ndmin=2).transpose())
        mmds.append(mmd)
        Z[sidx, :] = np.exp(map(posterior, x))
        sample.extend(list(n.simulate(len(sample), prngstate=prngstate)))
    return x, X, Y, Z, kl_divs, mmds, number_doublings, n, lowbound, posteriors


def fancy_kullback_leibler_graph((x, X, Y, Z, kl_divs, mmds, number_doublings,
                                  n, lowbound, posteriors)):
    """Display the convergence of the posterior of a NormalInvGammaPrior as it
    gets increasingly large samples from a given normal distribution.

    Not a very clear visualization for others -- too much information, I
    suppose.

    """
    import matplotlib.pyplot as plt
    # Although unused, this import registers the 3d projection with mpl
    from mpl_toolkits.mplot3d import Axes3D
    assert Axes3D  # silence flake8

    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the pdf of the actual distribution, at the back of the image
    ax.plot(x, len(x) * [number_doublings],
            np.exp(map(n.logpdf, x)), lw=5, color='r',
            label='Actual distribution')
    logbase = round((max(-log(abs(kl_divs))) / Z.max()) / log(10)) + 1
    base = int(10**logbase)
    ax.set_ylabel(
        'Slices are posterior distributions given sample of specified s\ize')
    ax.set_xlabel('Probability space (X)')
    ax.set_zlabel(
        'Probability density at X / $\log_{%i}$(|KL divergence|)' % base)
    ax.plot_wireframe(X, Y, Z)
    ax.set_yticklabels(['$2^{%i}$' % s for s in ax.get_yticks()])
    ax.plot(number_doublings * [lowbound - 1], xrange(number_doublings),
            -log(abs(kl_divs)) / log(base), lw=5, color='b',
            label='$-\log_{%i}$(|KL divergence|)' % base)
    plt.legend()
    plt.title('Convergence of NormalInvGammaPrior with increasing sample size')
    plt.ion()
    plt.show()
    return ax


def kullback_leibler_graph((x, X, Y, Z, kl_divs, mmds, number_doublings, n,
                            lowbound, posteriors)):
    """Since all of the pdfs are roughly normal, an improvement on the fancy graph
    might be to display summary statistics of them: the pdf values of actual
    mean and variance seems like a reasonable thing, along with KL divergence.

    """
    import matplotlib.pyplot as plt
    plt.ion()
    plt.clf()
    samplesizes = 2**scipy.arange(number_doublings)
    plt.plot(samplesizes, 1 / abs(kl_divs),
             label=('$1/D_{KL}(N(0,1) \| P(x|\{x_i\}_{i=1}^{n}\sim N(0,1), '
                    'NormalInvGammaPrior(0,1,1,1))$'))
    plt.plot(samplesizes, [exp(p.logpdf(Normal(0, 1))) for p in posteriors],
             label='P(N(0,1)|observed sample)')
    plt.plot(samplesizes, -log(mmds), label='mmd')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Size of sample used to calculate posterior distribution ($n$)')
    plt.ylabel('Posterior density of N(0,1) and 1 / KL(posterior||N(0,1))')
    plt.legend(loc='lower right')
    plt.title('Converging posterior of NormalInvGammaPrior as sample size '
              'increases')
    plt.gcf().set_size_inches(15, 10)
    plt.show()
