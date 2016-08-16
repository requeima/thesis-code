"""
Domain definitions.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from . import continuous
from reggie.core._singleton import singleton


__all__ = ['Box', 'Discrete', 'Grid', 'Sampled']


# DOMAIN BASE CLASS ###########################################################

class Domain(object):
    """
    Base class for domains over which an objective function should be
    optimized.
    """
    def init(self, rng=None, **kwargs):
        """
        Draw initial samples from the domain's input space using a domain-
        dependent strategy. The `rng` parameter should either be an instance of
        `numpy.RandomState` or an initial seed.
        """
        raise NotImplementedError

    def solve(self, f, rng=None, **kwargs):
        """
        Optimize the given index strategy f over the domain. If `X` is given
        this should be an array (or iterable) of initial points to begin
        optimization from.
        """
        raise NotImplementedError


# DOMAIN DEFINITIONS ##########################################################

class Box(Domain):
    """
    Domain which is defined by a set of upper and lower bounds in euclidean
    space.
    """
    def __init__(self, bounds):
        bounds = np.array(bounds, dtype=float, ndmin=2)
        if bounds.shape[1] != 2:
            raise ValueError('bounds must be a (d, 2)-array')
        self.bounds = bounds

    def init(self, method='latin', rng=None, **kwargs):
        return continuous.INITS[method](self.bounds, rng=rng, **kwargs)

    def solve(self, f, method='lbfgs', rng=None, **kwargs):
        return continuous.SOLVERS[method](f, self.bounds, rng=rng, **kwargs)


class Discrete(Domain):
    """
    An arbitrary discrete domain
    """
    def __init__(self, X):
        X = np.array(X, ndmin=1)

        # NOTE: this is a hack to make sure that for discrete grids over a
        # Euclidean input space we can just push these inputs into a GP object.
        if np.isrealobj(X) and X.ndim == 1:
            X = X[:, None]

        # save the array of potential inputs.
        self.X = X

    def solve(self, f, rng=None):
        F = f(self.X)
        i = F.argmax()
        return self.X[i], F[i]


class Grid(Discrete):
    """
    Discrete domain defined as a grid over some continuous space.
    """
    def __init__(self, bounds, n):
        bounds = np.array(bounds, dtype=float, ndmin=2)
        if bounds.shape[1] != 2:
            raise ValueError('bounds must be a (d, 2)-array')
        self.bounds = bounds

        # this creates a grid by dividing the bounds n times.
        X = np.vstack(_.flat for _ in
                      np.meshgrid(*[np.linspace(a, b, n)
                                    for (a, b) in bounds])).T
        super(Grid, self).__init__(X)

    def init(self, method='latin', rng=None, **kwargs):
        return continuous.INITS[method](self.bounds, rng=rng, **kwargs)


class Sampled(Domain):
    """
    A sampled discrete domain
    """

    def __init__(self, bounds, n):
        self.bounds = bounds
        self.n = n

    def X(self):
        bounds = np.array(self.bounds, dtype=float, ndmin=2)
        if bounds.shape[1] != 2:
            raise ValueError('bounds must be a (d, 2)-array')
        self.bounds = bounds
        self.ndim = bounds.shape[0]

        # sample from uniform distribution over the bounds
        X = np.empty((self.n, self.ndim))
        for d in range(self.ndim):
            X[:, d] = np.random.uniform(low=bounds[d][0], high=bounds[d][1], size=(self.n))

        if self.ndim == 1:
            X = np.sort(X, axis=0)


        X = np.array(X, ndmin=1)

        if np.isrealobj(X) and X.ndim == 1:
            X = X[:, None]

        # save the array of potential inputs.
        return X

    def solve(self, f, rng=None):
        X = self.X()
        F = f(X)
        i = F.argmax()
        return X[i], F[i]


    def init(self, method='latin', rng=None, **kwargs):
        return continuous.INITS[method](self.bounds, rng=rng, **kwargs)