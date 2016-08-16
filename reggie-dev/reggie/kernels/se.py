"""
Implementation of the squared-exponential kernels.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ..core.domains import Positive
from ..utils.misc import rstate

from ._core import RealKernel
from ._distances import rescale, dist, dist_foreach, diff

__all__ = ['SEARD']


class SEARD(RealKernel):
    """
    The squared-exponential kernel with ARD lengthscales ell and signal
    variance rho.
    """
    def __init__(self, rho, ell):
        super(SEARD, self).__init__(
            ('rho', rho, Positive()),
            ('ell', ell, Positive(), 'd'))

        # save the input dimension
        self.ndim = len(self._ell)

    def get_kernel(self, X1, X2=None):
        X1, X2 = rescale(self._ell, X1, X2)
        D = dist(X1, X2)
        K = self._rho * np.exp(-D/2)
        return K

    def get_dkernel(self, X1):
        return np.full(len(X1), self._rho)

    def get_grad(self, X1, X2=None):
        X1, X2 = rescale(self._ell, X1, X2)
        D = dist(X1, X2)
        E = np.exp(-D/2)
        K = self._rho * E
        G = np.empty((self.ndim+1,) + D.shape)
        G[0] = E
        for i, D in enumerate(dist_foreach(X1, X2)):
            G[i+1] = K * D / self._ell[i]
        return self._wrap_gradient(G)

    def get_dgrad(self, X1):
        G = np.zeros((self.ndim+1, len(X1)))
        G[0] = 1
        return self._wrap_gradient(G)

    def get_gradx(self, X1, X2=None):
        X1, X2 = rescale(self._ell, X1, X2)
        D = diff(X1, X2)
        K = self._rho * np.exp(-0.5 * np.sum(D**2, axis=-1))
        G = -K[:, :, None] * D / self._ell
        return G

    def get_dgradx(self, X1):
        return np.zeros_like(X1)

    def get_gradxy(self, X1, X2=None):
        X1, X2 = rescale(self._ell, X1, X2)
        D = diff(X1, X2)
        K = self._rho * np.exp(-0.5 * np.sum(D**2, axis=-1))
        D /= self._ell
        M = np.eye(self.ndim) / self._ell**2 - D[:, :, None] * D[:, :, :, None]
        G = M * K[:, :, None, None]
        return G

    def sample_spectrum(self, N, rng=None):
        rng = rstate(rng)
        W = rng.randn(N, self.ndim) / self._ell
        alpha = float(self._rho)
        return W, alpha
