"""
Implementation of the periodic kernels.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import autograd.numpy as np
from autograd import grad as grad
from autograd import jacobian


from ..core.domains import Positive
from ..utils.misc import rstate

from ._core import RealKernel
from ._distances import rescale, dist, dist_foreach, diff

__all__ = ['Periodic']

def kern_func(X1, X2, rho, ell, p):
    D = dist(X1, X2, 'euclidean') * np.pi / p
    K = (rho ** 2) * np.exp(-2 * (np.sin(D) / ell) ** 2)
    return K


class Periodic(RealKernel):
    """
    Covariance function for a 1-dimensional smooth periodic function with
    period p, lenthscale ell, and signal variance rho^2. The kernel function is
    given by::

        k(x, y) = rho^2 exp(-2 sin^2( ||x-y|| pi / p ) / ell^2)
    """
    def __init__(self, rho, ell, p):
        super(Periodic, self).__init__(
            ('rho', rho, Positive()),
            ('ell', ell, Positive(), 'd'),
            ('p', p, Positive()))

        self.ndim = len(self._ell)

    def get_kernel(self, X1, X2=None):
        D = dist(X1, X2, 'euclidean') * np.pi / self._p
        K = (self._rho**2) * np.exp(-2*(np.sin(D) / self._ell)**2)
        return K

    def get_dkernel(self, X1):
        return np.full(len(X1), self._rho**2)

    def get_grad(self, X1, X2=None):
        # # get the distance and a few transformations
        D = dist(X1, X2, 'euclidean') * np.pi / self._p
        G = np.empty((3,) + D.shape)

        kern = lambda rho: kern_func(X1, X2, rho, self._ell, self._p)
        G[0] = jacobian(kern)(self._rho)

        kern = lambda ell: kern_func(X1, X2, self._rho, ell, self._p)
        G[1] = jacobian(kern)(self._ell)[:,:,0]

        kern = lambda p: kern_func(X1, X2, self._rho, self._ell, p)
        G[2] = jacobian(kern)(self._p)

        return self._wrap_gradient(G)

    def get_dgrad(self, X1):
        G = np.zeros((3, len(X1)))
        G[0] = 2 * self._rho
        return self._wrap_gradient(G)

    def get_gradx(self, X1, X2=None):
        D = diff(X1, X2) * np.pi / self._p
        K = self._rho**2 * np.exp(-2*(np.sin(D) / self._ell)**2)
        G = -2 * np.pi / self._ell**2 / self._p * K * np.sin(2*D)
        return G


    def get_dgradx(self, X1):
        return np.zeros_like(X1)

    def get_gradxy(self, X1, X2=None):
        def kern(X1, X2):
            D =  np.linalg.norm(X1  - X2 )* np.pi / self._p
            K = (self._rho ** 2) * np.exp(-2 * (np.sin(D) / self._ell) ** 2)
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


    def sample_spectrum(self, N, rng=None):
        raise NotImplementedError
