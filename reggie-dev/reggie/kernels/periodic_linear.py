"""
Implementation of the periodic kernels.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# import numpy as np

from ..core.domains import Positive, Real
from ..utils.misc import rstate

from ._core import RealKernel
from ._distances import rescale, dist, dist_foreach, diff

import numpy as np
import autograd.numpy as np
from autograd import grad
from autograd import jacobian



__all__ = ['PeriodicLinear']


def per_kern(X1, X2, rho, ell, p):
    D = dist(X1, X2, 'euclidean') * np.pi / p
    K = (rho ** 2) * np.exp(-2 * (np.sin(D) / ell) ** 2)
    return K


class PeriodicLinear(RealKernel):
    """
    Periodic plus linear kernel
    """
    def __init__(self, rho_p, ell, p, rho_v, rho_b, c):
        super(PeriodicLinear, self).__init__(
            ('rho_p', rho_p, Positive()),
            ('ell', ell, Positive()),
            ('p', p, Positive()),
            ('rho_b', rho_b, Positive()),
            ('rho_v', rho_v, Positive()),
            ('c', c, Real(), 'd'))

        self.ndim = len(self._c)

    def get_kernel(self, X1, X2=None):

        D = dist(X1, X2, 'euclidean') * np.pi / self._p
        K = (self._rho_p ** 2) * np.exp(-2 * (np.sin(D) / self._ell) ** 2)

        L = self._rho_b**2 + self._rho_v**2 * np.sum((X1 - self._c )[:, None, :] * (X2 - self._c)[None, :, :], axis=2)

        K += L
        return K

    def get_dkernel(self, X1):
        return self.get_kernel(X1).diagonal().copy()

    def get_grad(self, X1, X2=None):
        # get the distance and a few transformations
        D = dist(X1, X2, 'euclidean') * np.pi / self._p
        G = np.empty((5 + self.ndim,) + D.shape)

        kern = lambda rho: per_kern(X1, X2, rho, self._ell, self._p)
        G[0] = jacobian(kern)(self._rho_p)

        kern = lambda ell: per_kern(X1, X2, self._rho_p, ell, self._p)
        G[1] = jacobian(kern)(self._ell)

        kern = lambda p: per_kern(X1, X2, self._rho_p, self._ell, p)
        G[2] = jacobian(kern)(self._p)

        G[3] = 2*self._rho_b * np.ones_like(D)
        G[4] = 2 * self._rho_v*np.sum((X1 - self._c )[:, None, :] * (X2 - self._c)[None, :, :], axis=2)
        G[5:5 + self.ndim] =  self._rho_v**2 * np.transpose(2 * self._c[None, None, :] - X1[:, None, :] - X2[None, :, :], axes=(2,0, 1))
        return self._wrap_gradient(G)

    def get_dgrad(self, X1):
        return np.vstack(map(np.diag, self.get_grad(X1))).copy()

    def get_gradx(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        D = diff(X1, X2) * np.pi / self._p
        K = self._rho_p ** 2 * np.exp(-2 * (np.sin(D) / self._ell) ** 2)
        G = -2 * np.pi / self._ell**2 / self._p * K * np.sin(2*D)
        # add linear gradient
        G += self._rho_v**2 * (X2 - self._c)[None, :]
        return G

    def get_dgradx(self, X1):
        G = self.get_gradx(X1)
        G = np.vstack(G[i, i] for i in xrange(len(G)))
        return G

    def get_gradxy(self, X1, X2=None):
        def kern(X1, X2):
            D =  np.linalg.norm(X1  - X2 )* np.pi / self._p
            K = (self._rho_p ** 2) * np.exp(-2 * (np.sin(D) / self._ell) ** 2)
            L = self._rho_b ** 2 + self._rho_v ** 2 * np.sum((X1 - self._c) * (X2 - self._c))
            return K + L


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


    def sample_spectrum(self, N, rng=None):
        raise NotImplementedError
