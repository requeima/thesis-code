"""
Definition of the likelihood interface.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from ..core.params import Parameterized

__all__ = ['Likelihood']


### BASE LIKELIHOOD INTERFACE #################################################

class Likelihood(Parameterized):
    """
    The base Likelihood interface.
    """
    def get_variance(self):
        """
        Return the variance of the observation model; this is used for
        performing exact inference and should only be implemented by Gaussian
        models.
        """
        raise NotImplementedError

    def predict(self, mu, s2):
        """
        Return predictions given values for the mean and variance of the latent
        function f.
        """
        raise NotImplementedError

    def sample(self, f, rng=None):
        """
        Sample observations y given evaluations of the latent function f.
        """
        raise NotImplementedError

    def get_loglike(self, y, f):
        """
        Get the marginal log-likelihood, i.e. log p(y|f), along with the first
        three derivatives of this quantity wrt f; returns a 4-tuple.
        """
        raise NotImplementedError

    def get_laplace_grad(self, y, f):
        """
        Get the gradients necessary to compute a Laplace approximation. This
        should return, for each likelihood parameter, a 3-tuple containing the
        derivatives::

            d   log p(y|f) / dtheta_i
            d^2 log p(y|f) / dtheta_i df
            d^3 log p(y|f) / dtheta_i df^2

        with respect to the ith likelihood parameter and the latent function.
        This should return an array of size (len(self.hyper), 3, len(y)).
        """
        raise NotImplementedError

    def get_tail(self, f, mu, s2, dmu=None, ds2=None):
        """
        Return the probability that inputs with the given latent mean `mu` and
        variance `s2` exceed the target value `f`. If `dmu` and `ds2` are not
        `None` then return the derivatives of this quantity with respect to the
        input space.
        """
        raise NotImplementedError

    def get_improvement(self, f, mu, s2, dmu=None, ds2=None):
        """
        Return the expected improvement of inputs with the given latent mean
        `mu` and variance `s2` exceed the target value `f`. If `dmu` and `ds2`
        are not `None` then return the derivatives of this quantity with
        respect to the input space.
        """
        raise NotImplementedError

    def get_entropy(self, mu, s2, dmu=None, ds2=None):
        """
        Return the predictive entropy of inputs with the given latent mean `mu`
        and variance `s2`. If `dmu` and `ds2` are not `None` then return the
        derivatives of this quantity with respect to the input space.
        """
        raise NotImplementedError
