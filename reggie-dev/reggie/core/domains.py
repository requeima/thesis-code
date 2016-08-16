"""
Domains and domain transformations.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ._singleton import Singleton, singleton
from ..utils.misc import create_setstate, create_getstate

__all__ = ['Real', 'Positive', 'Bounded']


# numerical constants
epsilon = np.finfo(np.float64).resolution


class Transform(Singleton):
    """
    Interface for parameter transformations.
    """
    __slots__ = []
    __getstate__ = create_getstate(__slots__)
    __setstate__ = create_setstate(__slots__)

    def get_transform(self, x):
        """
        Transform a parameter with value `x` from its original space into the
        transformed space `f(x)`.
        """
        raise NotImplementedError

    def get_gradfactor(self, x):
        """
        Get the gradient factor for the transformation. This computes and
        returns the gradient of `f^{-1}(t)` evaluated at `t=f(x)`.
        """
        raise NotImplementedError

    def get_inverse(self, t):
        """
        Apply the inverse transformation which takes a transformed parameter
        `t=f(x)` and returns the original value `x`.
        """
        raise NotImplementedError

    def get_image(self, domain):
        """
        Get the image of a given domain under this transformation.
        """
        raise NotImplementedError


class LogTransform(Transform):
    """
    Log transform.
    """
    __slots__ = []
    __getstate__ = create_getstate(__slots__)
    __setstate__ = create_setstate(__slots__)

    def get_transform(self, x):
        return np.log(x)

    def get_gradfactor(self, x):
        return x

    def get_inverse(self, t):
        return np.clip(np.exp(t), epsilon, np.inf)

    def get_image(self, domain):
        if domain is Positive():
            return Real()
        elif isinstance(domain, Bounded) and domain.bounds[0] > 0:
            return Bounded(self.get_transform(domain.bounds[0]),
                           self.get_transform(domain.bounds[1]))
        else:
            raise ValueError('the LogTransform is not defined for this domain')


class Identity(Transform):
    """
    Identity transform.
    """
    __slots__ = []
    __getstate__ = create_getstate(__slots__)
    __setstate__ = create_setstate(__slots__)

    def get_transform(self, x):
        return x

    def get_gradfactor(self, x):
        return 1.0

    def get_inverse(self, t):
        return t

    def get_image(self, domain):
        return domain


class Domain(Singleton):
    """
    Definition of a domain.
    """
    __slots__ = []
    __getstate__ = create_getstate(__slots__)
    __setstate__ = create_setstate(__slots__)

    transform = Identity()
    bounds = (None, None)

    def __le__(self, other):
        raise NotImplementedError

    def __contains__(self, item):
        raise NotImplementedError

    def project(self, x):
        """
        Project the value x onto the space of the domain.
        """
        raise NotImplementedError


class Real(Domain):
    """
    Domain for real-valued parameters.
    """
    __slots__ = []
    __getstate__ = create_getstate(__slots__)
    __setstate__ = create_setstate(__slots__)

    def __le__(self, other):
        return other is Real()

    def __contains__(self, item):
        return True

    def project(self, x):
        return x


class Positive(Domain):
    """
    Domain for positive-valued parameters.
    """
    __slots__ = []
    __getstate__ = create_getstate(__slots__)
    __setstate__ = create_setstate(__slots__)

    transform = LogTransform()
    bounds = (epsilon, None)

    def __le__(self, other):
        return (other is Real()) or (other is Positive())

    def __contains__(self, item):
        return item > 0

    def project(self, x):
        return np.clip(x, epsilon, np.inf)


class Bounded(Domain):
    """
    Domain for parameters whose values are bounded by finite values [a, b] for
    values of a not equal to b.
    """
    __slots__ = ['bounds']
    __getstate__ = create_getstate(__slots__)
    __setstate__ = create_setstate(__slots__)

    __types__ = 'ff'

    def _init(self, a, b):
        # pylint: disable=C0111
        if b <= a or not all(np.isfinite((a, b))):
            raise ValueError("malformed upper/lower bounds")
        self.bounds = (a, b)

    def __le__(self, other):
        return (other is Real()) or (other is Positive() and self.bounds[0] > 0)

    def __contains__(self, item):
        return self.bounds[0] <= item and item <= self.bounds[1]

    def project(self, x):
        return np.clip(x, self.bounds[0], self.bounds[1])
