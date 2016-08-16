"""
Implementation of the matern kernel.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ..core.domains import Positive
from ..utils.misc import rstate

from ._core import RealKernel
from ._distances import rescale, dist, dist_foreach, diff

__all__ = ['MaternARD']


_F = {}
_F[1] = lambda _: 1
_F[3] = lambda r: 1 + r
_F[5] = lambda r: 1 + r * (1 + r/3.)

_G = {}
_G[1] = lambda _: 1
_G[3] = lambda r: r
_G[5] = lambda r: r * (1+r) / 3.


class MaternARD(RealKernel):
    """
    The Matern kernel with ARD lengthscales ell and signal variance rho.
    """
    def __init__(self, rho, ell, d=3):
        if d not in {1, 3, 5}:
            raise ValueError('d must be one of 1, 3, or 5')

        super(MaternARD, self).__init__(
            ('rho', rho, Positive()),
            ('ell', ell, Positive(), 'd'), d=d)

        # save the input dimension
        self.ndim = len(self._ell)

        # save the type of Matern kernel.
        self._d = d
        self._f = _F[d]
        self._g = _G[d]

    def get_kernel(self, X1, X2=None):
        X1, X2 = rescale(self._ell / np.sqrt(self._d), X1, X2)
        D = dist(X1, X2, metric='euclidean')
        K = self._rho * np.exp(-D) * self._f(D)
        return K

    def get_dkernel(self, X1):
        return np.full(len(X1), self._rho)

    def get_grad(self, X1, X2=None):
        X1, X2 = rescale(self._ell / np.sqrt(self._d), X1, X2)
        D = dist(X1, X2, metric='euclidean')
        E = np.exp(-D)
        S = E * self._f(D)
        M = self._rho * E * self._g(D)
        G = np.empty((self.ndim+1,) + D.shape)
        G[0] = S
        for i, D_ in enumerate(dist_foreach(X1, X2)):
            with np.errstate(invalid='ignore'):
                G[i+1] = np.where(D < 1e-12, 0, M * D_ / D / self._ell[i])
        return self._wrap_gradient(G)

    def get_dgrad(self, X1):
        G = np.zeros((self.ndim+1, len(X1)))
        G[0] = 1
        return self._wrap_gradient(G)

    def get_gradx(self, X1, X2=None):
        ell = self._ell / np.sqrt(self._d)
        X1, X2 = rescale(ell, X1, X2)
        D1 = diff(X1, X2)
        D = np.sqrt(np.sum(D1**2, axis=-1))
        S = self._rho * np.exp(-D)
        with np.errstate(invalid='ignore', divide='ignore'):
            M = np.where(D < 1e-12, 0, S * self._g(D) / D)
        G = -M[:, :, None] * D1 / ell
        return G

    def get_dgradx(self, X1):
        return np.zeros_like(X1)

    def get_gradxy(self, X1, X2=None):
        raise NotImplementedError

    def sample_spectrum(self, N, rng=None):
        rng = rstate(rng)
        g = np.sqrt(rng.gamma(self._d / 2., 2. / self._d, N))
        W = rng.randn(N, self.ndim) / self._ell / g[:, None]
        a = float(self._rho)
        return W, a
