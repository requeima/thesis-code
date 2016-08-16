"""
Definitions of various priors. All priors defined here are independently
defined over a single scalar random variable.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ..utils.misc import rstate
from ._singleton import Singleton
from . import domains
from ..utils.misc import create_setstate, create_getstate


__all__ = ['Uniform', 'Normal', 'LogNormal', 'Horseshoe']

# numerical constants
log2pi = np.log(2*np.pi)


class Prior(Singleton):
    """
    Interface for prior distributions. Priors act as parameterized singletons.
    For the most part this can be ignored except for the fact that `_init`
    should be used in place of the standard initialization.
    """
    __slots__ = []
    __getstate__ = create_getstate(__slots__)
    __setstate__ = create_setstate(__slots__)

    def sample(self, size=None, rng=None):
        """
        Sample from the prior. If `size` is None return a single scalar sample,
        otherwise return `size` array of samples.
        """
        raise NotImplementedError

    def get_logprior(self, theta, grad=False):
        """
        Compute the log prior evaluated at the parameter `theta`. If
        `grad` is True return the derivative of this quantity wrt `theta`.
        """
        raise NotImplementedError


class Uniform(Prior):
    """
    Uniform prior over the range [a, b].
    """
    __slots__ = ['_a', '_b']
    __getstate__ = create_getstate(__slots__)
    __setstate__ = create_setstate(__slots__)
    __types__ = 'ff'

    def _init(self, a, b):
        # pylint: disable=C0111
        if b <= a or not all(np.isfinite((a, b))):
            raise ValueError("malformed upper/lower bounds")
        self._a = a
        self._b = b

    @property
    def domain(self):
        """
        Return the new domain.
        """
        return domains.Bounded(self._a, self._b)

    def sample(self, size=None, rng=None):
        rng = rstate(rng)
        rnd = rng.rand() if (size is None) else rng.rand(size)
        return self._a + (self._b - self._a) * rnd

    def get_logprior(self, theta, grad=False):
        if self._a <= theta and theta <= self._b:
            return (0.0, 0.0) if grad else 0.0
        else:
            return (-np.inf, 0.0) if grad else -np.inf


class Normal(Prior):
    """
    Normal prior where with mean `mu` and variance `s2`.
    """
    __slots__ = ['_mu', '_sigma', '_logsigma']
    __getstate__ = create_getstate(__slots__)
    __setstate__ = create_setstate(__slots__)
    __types__ = 'ff'

    domain = domains.Real()

    def _init(self, mu, s2):
        # pylint: disable=C0111
        self._mu = mu
        self._sigma = np.sqrt(s2)
        self._logsigma = np.log(self._sigma)

    def sample(self, size=None, rng=None):
        return rstate(rng).normal(self._mu, self._sigma, size)

    def get_logprior(self, theta, grad=False):
        standard = (theta - self._mu) / self._sigma
        logp = -0.5*log2pi - self._logsigma - 0.5*standard**2
        dlogp = -standard / self._sigma
        return (logp, dlogp) if grad else logp


class LogNormal(Prior):
    """
    Log-normal prior with mean `mu` and variance `s2`.
    """
    __slots__ = ['_mu', '_sigma', '_logsigma']
    __getstate__ = create_getstate(__slots__)
    __setstate__ = create_setstate(__slots__)
    __types__ = 'ff'

    domain = domains.Positive()

    def _init(self, mu, s2):
        # pylint: disable=C0111
        self._mu = mu
        self._sigma = np.sqrt(s2)
        self._logsigma = np.log(self._sigma)

    def sample(self, size=None, rng=None):
        return rstate(rng).lognormal(self._mu, self._sigma, size)

    def get_logprior(self, theta, grad=False):
        logtheta = np.log(theta)
        standard = (logtheta - self._mu) / self._sigma
        logp = -logtheta - self._logsigma - 0.5*log2pi - 0.5*standard**2
        dlogp = -(standard / self._sigma + 1) / theta
        return (logp, dlogp) if grad else logp


class Horseshoe(Prior):
    """
    Horseshoe prior with a given `scale` parameter.
    """
    __slots__ = ['_scale']
    __getstate__ = create_getstate(__slots__)
    __setstate__ = create_setstate(__slots__)

    __types__ = 'f'
    domain = domains.Positive()

    def _init(self, scale):
        # pylint: disable=C0111
        self._scale = scale

    def get_logprior(self, theta, grad=False):
        theta2_inv = (self._scale / theta)**2
        inner = np.log1p(theta2_inv)
        logp = np.sum(np.log(inner))
        dlogp = -2 * (theta2_inv / (1 + theta2_inv)) / inner / theta
        return (logp, dlogp) if grad else logp
