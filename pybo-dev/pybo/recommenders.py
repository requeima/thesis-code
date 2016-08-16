"""
Recommendations.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

__all__ = ['best_latent', 'best_incumbent']


def best_latent(model, domain, X, Y, rng=None):
    """
    Given a model return the best recommendation, corresponding to the point
    with maximum posterior mean.
    """
    def mu(X, grad=False):
        """Posterior mean objective function."""
        if grad:
            return model.predict(X, True)[::2]
        else:
            return model.predict(X)[0]

    # maximize the mean over the domain
    xbest, _ = domain.solve(mu, X)
    
    return xbest


def best_incumbent(model, domain, X, Y, rng=None):
    """
    Return a recommendation given by the best latent function value evaluated
    at points seen so far.
    """
    f, _ = model.predict(X)
    return X[f.argmax()]


def best_observed(model, domain, X, Y, rng=None):
    """
    Return a recommendation given by the best input corresponding to the best
    observed function value seen so far.
    """
    return X[np.argmax(Y)]
