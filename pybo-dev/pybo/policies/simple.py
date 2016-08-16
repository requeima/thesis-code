"""
Acquisition functions based on the probability or expected value of
improvement.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

__all__ = ['EI', 'PI', 'UCB', 'Thompson', 'PES']


def EI(model, domain, X, rng, xi=0.0):
    """
    Expected improvement policy with an exploration parameter of `xi`.
    """
    model = model.copy()
    target = model.predict(X)[0].max() + xi

    def index(X, grad=False):
        """EI policy instance."""
        return model.get_improvement(target, X, grad)

    return index


def PI(model, domain, X, rng, xi=0.05):
    """
    Probability of improvement policy with an exploration parameter of `xi`.
    """
    model = model.copy()
    target = model.predict(X)[0].max() + xi

    def index(X, grad=False):
        """PI policy instance."""
        return model.get_tail(target, X, grad)

    return index


def Thompson(model, domain, X, rng, n=100):
    """
    Thompson sampling policy.
    """
    return model.sample_f(n, rng).get


def UCB(model, domain, X, rng, delta=0.1, xi=0.2):
    """
    The (GP)UCB acquisition function where `delta` is the probability that the
    upper bound holds and `xi` is a multiplicative modification of the
    exploration factor.
    """
    model = model.copy()
    d = len(X)
    a = xi * 2 * np.log(np.pi**2 / 3 / delta)
    b = xi * (4 + d)

    def index(X, grad=False):
        """UCB policy instance."""
        posterior = model.predict(X, grad=grad)
        mu, s2 = posterior[:2]
        beta = a + b * np.log(d+1)
        if grad:
            dmu, ds2 = posterior[2:]
            return (mu + np.sqrt(beta * s2),
                    dmu + 0.5 * np.sqrt(beta / s2[:, None]) * ds2)
        else:
            return mu + np.sqrt(beta * s2)

    return index

def _gmm_entropy_integrand(y, mu, s2):
    """
    Helper function - returns the density of a gmm with means in vector
    mu and variances in vector s2 at the point y
    """
    #log-sum-exp trick for computing log of the gmm pdf
    log_gmm = np.array(norm.logpdf(y, loc=mu, scale=np.sqrt(s2)))
    max_log = np.max(log_gmm)
    sum_log = np.sum(np.exp(log_gmm - max_log))
    log_p = max_log + np.log(sum_log) - np.log(len(log_gmm)) # subtract dividing constant in mean
    p = np.exp(log_p)
    return -p*log_p

def PES(model, domain, X, rng, nopt=1, nfeat=300, opes=True, ipes=False):
    model = model.copy()

    # construct a list of model/function tuples
    if hasattr(model, '__iter__'):
        samples = [(m, m.sample_f(nfeat, rng)) for _ in xrange(nopt) for m in model]
        # samples = [(m, m.sample_f(nfeat, rng)) for m in model]

    else:
        samples = [(model, model.sample_f(nfeat, rng)) for _ in xrange(nopt)]

    # construct model/xstar/fstar tuples given the function sample
    samples = [(m,) + domain.solve(f.get) for (m, f) in samples]

    # construct the posterior either by conditioning on the maximizer or its
    # value based on the opes flag
    post = [m.condition_fstar(fstar) if opes else
            m.condition_xstar(xstar)
            for (m, xstar, fstar) in samples]

    def index(X, grad=False):
        if ipes:
            parts = np.array([_.predict(X) for _ in post])
            mu = parts[:, 0, :]
            s2 = parts[:, 1, :]
            like = np.array([_._like.get_variance() for _ in post])
            s2 += np.tile(np.reshape(like, (s2.shape[0], 1)), (1, s2.shape[1]))
            n = len(X)
            H1 = model.get_ipes_entropy(X)
            H2 = np.array([quad(_gmm_entropy_integrand,
                                np.min(mu[:, i]) - 100.0*np.max(s2[:, i]), #lower integration bound
                                np.max(mu[:, i]) + 100.0*np.max(s2[:, i]), #upper integration bound
                                args=(mu[:,i], s2[:, i]))[0] for i in range(n)])
            H = H1 - H2
            return H
        else:
            if not grad:
                H1 = model.get_entropy(X)
                H2 = np.mean([_.get_entropy(X) for _ in post], axis=0)
                H = H1 - H2
                return H
            else:
                H, dH = model.get_entropy(X, grad=True)
                parts = zip(*[_.get_entropy(X, True) for _ in post])
                H -= np.mean(parts[0], axis=0)
                dH -= np.mean(parts[1], axis=0)
                return H, dH

    return index
