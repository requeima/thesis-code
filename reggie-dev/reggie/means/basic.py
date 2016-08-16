"""
Implementation of basic functions.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ._core import Mean, RealMean
from ..core.domains import Real

__all__ = ['Zero', 'Constant', 'Linear']


class Zero(Mean):
    """
    Function which returns zero on any input. Note that because this mean is a
    constant we can compute the gradient with respect to inputs X (it is always
    zero) if the input space is continuous although this need not be the case.
    """
    def __init__(self):
        super(Zero, self).__init__()

    def get_mean(self, X):
        return np.zeros(len(X))

    def get_grad(self, X):
        return np.zeros((0, len(X)))

    def get_gradx(self, X):
        """
        Return the gradient with respect to the input space.
        """
        return np.zeros_like(X)


class Constant(Mean):
    """
    Function which returns a constant value on any input. Note that because
    this mean is a constant we can compute the gradient with respect to inputs
    X (it is always zero) if the input space is continuous although this need
    not be the case.
    """
    def __init__(self, bias=0):
        super(Constant, self).__init__(
            ('bias', bias, Real()))

    def get_mean(self, X):
        return np.full(len(X), self._bias)

    def get_grad(self, X):
        return self._wrap_gradient(np.ones((1, len(X))))

    def get_gradx(self, X):
        """
        Return the gradient with respect to the input space.
        """
        return np.zeros_like(X)


class Linear(RealMean):
    """
    Linear mean function. The function is defined by::

        f(x) = x' * theta + bias

    where `theta` defines a vector of slopes of the same dimensionality as x
    and where `bias` should be a constant mean value.
    """
    def __init__(self, theta, bias=0):
        super(Linear, self).__init__(
            ('theta', theta, Real(), 'd'),
            ('bias', bias, Real()))

        # save the input dimensions
        self.ndim = len(self._theta)

    def get_mean(self, X):
        return np.dot(X, self._theta) + self._bias

    def get_grad(self, X):
        return self._wrap_gradient(np.r_[X.copy().T, np.ones((1, len(X)))])

    def get_gradx(self, X):
        return np.tile(self._theta, (len(X), 1))
