"""
Inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from ... import likelihoods
from ... import kernels
from ... import means

from .bootsample import BootStrapSample
from .gp import GP

__all__ = ['BootGP', 'make_bootgp']


class BootGP(GP):
    def __init__(self, like, kern, mean, inf='exact', U=None):
        super(BootGP, self).__init__(like, kern, mean, inf, U)

    def sample_f(self, n, rng=None):
        return BootStrapSample(self, n, rng)


def make_bootgp(sn2, rho, ell, mean=0.0, p=None, kernel='se', inf='exact', U=None):
    """
    Simple interface for creating a GP.
    """
    # create the mean/likelihood objects
    like = likelihoods.Gaussian(sn2)
    mean = means.Constant(mean)

    # create a kernel object which depends on the string identifier
    kern = (
        kernels.SEARD(rho, ell) if (kernel == 'se') else
        kernels.MaternARD(rho, ell, 1) if (kernel == 'matern1') else
        kernels.MaternARD(rho, ell, 3) if (kernel == 'matern3') else
        kernels.MaternARD(rho, ell, 5) if (kernel == 'matern5') else
        kernels.Periodic(rho, ell, p) if (kernel == 'periodic') else
        None)

    if kernel is None:
        raise ValueError('Unknown kernel type')

    return BootGP(like, kern, mean, inf, U)