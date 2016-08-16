"""
Implementation of the squared-exponential kernels.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import autograd.numpy as np
from autograd import grad as grad
from autograd import jacobian

from scipy.stats import beta as beta_dist
from scipy.special import beta as beta_func
import scipy.optimize as spop


from ..core.domains import Positive
from ..utils.misc import rstate

from ._core import RealKernel
from ._distances import rescale, dist_foreach, diff

__all__ = ['WARPSEARD']


def dist(X1, X2):
    try:
        D = np.sum(np.square(X1[:, None, :] - X2[None, :, :]), axis=2)
    except:
        D = np.sum(np.square(X1[:, None] - X2[None, :]), axis=1)
    return D


class WARPSEARD(RealKernel):
    """
    The squared-exponential kernel with ARD lengthscales ell and signal
    variance rho.
    """
    def __init__(self, rho, ell, alpha, beta, bounds):
        super(WARPSEARD, self).__init__(
            ('rho', rho, Positive()),
            ('ell', ell, Positive(), 'd'),
            ('alpha', alpha, Positive(), 'd'),
            ('beta', beta, Positive(), 'd'))

        # save the input dimension
        self.ndim = len(self._ell)
        self.bounds = np.array(bounds, dtype=float, ndmin=2)
        self.warped = True

    def warp_input(self, X, alpha=None, beta=None):
        bounds = np.array(self.bounds)
        if alpha is None:
            alpha = self._alpha
        if beta is None:
            beta = self._beta
        if X is None:
            return None

        X = np.array(X)
        X_warped = np.empty((X).shape)
        for n in range(self.ndim):
            # a hack way to deal with the numpy shapes problem. This should be fixed
            try:

                X_warped[:, n:n+1] = (X[:, n:n+1] - bounds[n, 0])/(bounds[n, 1] - bounds[n, 0])
                # use beta CDF warping
                X_warped[:, n:n+1] = beta_dist.cdf(X_warped[:, n:n+1], alpha[n], beta[n])
                X_warped = (bounds[n, 1] - bounds[n, 0])*X_warped + bounds[n, 0]
            except:
                X_warped = (X - bounds[n, 0]) / (bounds[n, 1] - bounds[n, 0])
                # use beta CDF warping
                X_warped[:] = beta_dist.cdf(X_warped[:], alpha[n], beta[n])
                X_warped = (bounds[n, 1] - bounds[n, 0]) * X_warped + bounds[n, 0]
        return X_warped

    def get_kernel(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        X1 = self.warp_input(X1)
        X2 = self.warp_input(X2)
        X1, X2 = rescale(self._ell, X1, X2)
        D = dist(X1, X2)
        K = self._rho * np.exp(-D/2)
        return K

    def get_dkernel(self, X1):
        return self.get_kernel(X1).diagonal().copy()


    def sample_spectrum(self, N, rng=None):
        rng = rstate(rng)
        W = rng.randn(N, self.ndim) / self._ell
        alpha = float(self._rho)
        return W, alpha


    def get_gradx(self, X1, X2=None):
        def kern(X1, X2):
            if X2 is None:
                X2 = X1

            X1 = self.warp_input(X1, self._alpha, self._beta)
            X2 = self.warp_input(X2, self._alpha, self._beta)
            X1, X2 = rescale(self._ell, X1, X2)
            D = dist(X1, X2)
            K = self._rho * np.exp(-D / 2)
            return K

        if X2 is None:
            X2 = X1

        grad1 = grad(kern)
        n = len(X1)
        m = len(X2)
        G = np.empty((n, m, self.ndim))

        for i in range(n):
            for j in range(m):
                for d in range(self.ndim):
                    G[i, j, d] = grad1(X1[i], X2[j])[d]
        return G


    def get_gradxy(self, X1, X2=None):
            def kern(X1, X2):
                if X2 is None:
                    X2 = X1

                X1 = self.warp_input(X1, self._alpha, self._beta)
                X2 = self.warp_input(X2, self._alpha, self._beta)
                X1, X2 = rescale(self._ell, X1, X2)
                D = dist(X1, X2)

                K = self._rho * np.exp(-D / 2)
                return K

            if X2 is None:
                X2 = X1

            grad1 = grad(kern)
            n = len(X1)
            m = len(X2)
            G = np.empty((n, m, self.ndim, self.ndim))

            for i in range(n):
                for d in range(self.ndim):
                    dkern = lambda Y: grad1(X1[i], Y)[d]
                    for j in range(m):
                        grad2 = grad(dkern)
                        G[i, j, d, :] = grad2(X2[j])
            return G
