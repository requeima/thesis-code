"""
Inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import warnings
import inspect

from ...utils.misc import rstate
from ...utils import linalg as la

from ... import likelihoods
from ... import kernels
from ... import means

from .._core import ParameterizedModel
from . import inference

from .fourier import FourierSample
from .bootsample import BootStrapSample
from .conditional import GP_fstar, GP_xstar

__all__ = ['GP', 'make_gp']


class GP(ParameterizedModel):
    """
    Implementation of GP inference.
    """
    def __init__(self, like, kern, mean, inf='exact', U=None):
        # translate the inference string into a function
        if isinstance(inf, basestring):
            if inf in inference.__all__:
                infer = getattr(inference, inf)
            else:
                raise ValueError('Unknown inference method')

        # do some light error-checking to make sure that if we pass inducing
        # points that they are actually valid for this inference method.
        if U is not None and len(inspect.getargspec(infer).args) != 6:
            raise ValueError('the given inference method does not support '
                             'inducing points')

        # This will define the type necessary for the like parameter. If the
        # inference method given corresponds to an "exact" method then we will
        # require a Gaussian likelihood. Otherwise the base Likelihood type is
        # acceptable. This check is performed inside Parameterized.__init__.
        LikelihoodType = (likelihoods.Gaussian if inf in ('fitc', 'exact') else
                          likelihoods.Likelihood)

        # initialize
        super(GP, self).__init__(
            ('like', like, LikelihoodType),
            ('kern', kern, kernels.Kernel),
            ('mean', mean, means.Mean), inf=inf)

        # this is a non-parametric model so we'll need to store the data.
        self._X = None
        self._Y = None
        self._U = U
        self.ndim = kern.ndim

        # store the inference method which should just be a function as well as
        # the posterior sufficient statistics (None so far).
        self._infer = infer
        self._post = None

    def __deepcopy__(self, memo):
        # don't make a copy of the data.
        memo[id(self._X)] = self._X
        memo[id(self._Y)] = self._Y
        return super(GP, self).__deepcopy__(memo)

    def add_data(self, X, Y):
        X = np.array(X, copy=False, ndmin=2, dtype=float)
        Y = np.array(Y, copy=False, ndmin=1, dtype=float)
        if self._X is None:
            self._X = X.copy()
            self._Y = Y.copy()
        else:
            self._X = np.r_[self._X, X]
            self._Y = np.r_[self._Y, Y]
        self._update()

    def _update(self):
        # NOTE: this method is called both after adding data as well as any
        # time that the parameters change.
        if self._X is None:
            self._post = None
        else:
            args = (self._like, self._kern, self._mean, self._X, self._Y)
            if self._U is not None:
                args += (self._U, )
            self._post = self._infer(*args)

    def _predict(self, X, joint=False, grad=False):
        """
        Internal method used to make both joint and marginal predictions.
        """
        # get the prior mean and variance
        mu = self._mean.get_mean(X)
        s2 = (self._kern.get_kernel(X) if joint else
              self._kern.get_dkernel(X))

        # if we have data compute the posterior
        if self._post is not None:
            if self._U is not None:
                K = self._kern.get_kernel(self._U, X)
            else:
                K = self._kern.get_kernel(self._X, X)

            # compute the mean and variance
            w = self._post.w.reshape(-1, 1)
            V = la.solve_triangular(self._post.L, w*K)
            mu += np.dot(K.T, self._post.a)
            s2 -= np.dot(V.T, V) if joint else np.sum(V**2, axis=0).copy()

            # add on a correction factor if necessary
            if self._post.C is not None:
                VC = la.solve_triangular(self._post.C, K)
                s2 += np.dot(VC.T, VC) if joint else np.sum(VC**2, axis=0)

        # FIXME: make sure s2 isn't zero. this is almost equivalent to using a
        # nugget parameter, but after the fact if the predictive variance is
        # too small.
        if not joint:
            s2 = np.clip(s2, 1e-100, np.inf)

        if not grad:
            return mu, s2

        if joint:
            raise ValueError('cannot compute gradients of joint predictions')

        dmu = self._mean.get_gradx(X)
        ds2 = self._kern.get_dgradx(X)

        if self._post is not None:
            # get the kernel gradients
            if self._U is not None:
                dK = self._kern.get_gradx(X, self._U)
            else:
                dK = self._kern.get_gradx(X, self._X)

            # reshape them to make it a 2d-array
            dK = np.rollaxis(dK, 1)
            dK = np.reshape(dK, (dK.shape[0], -1))

            # compute the mean gradients
            dmu += np.dot(dK.T, self._post.a).reshape(X.shape)

            # compute the variance gradients
            dV = la.solve_triangular(self._post.L, w*dK)
            dV = np.rollaxis(np.reshape(dV, (-1,) + X.shape), 2)
            ds2 -= 2 * np.sum(dV * V, axis=1).T

            # add in a correction factor
            if self._post.C is not None:
                dVC = la.solve_triangular(self._post.C, dK)
                dVC = np.rollaxis(np.reshape(dVC, (-1,) + X.shape), 2)
                ds2 += 2 * np.sum(dVC * VC, axis=1).T

        return mu, s2, dmu, ds2

    def get_loglike(self, grad=False):
        if self._post is None:
            # if we haven't added any data then the posterior parameters
            # shouldn't exist and we will define the loglikelihood to be 0.0
            loglike = 0.0
            dloglike = np.zeros(self._nhyper)
        else:
            # otherwise it should be computed by the inference procedure
            # already and we can just read it out.
            loglike = self._post.lZ
            dloglike = self._post.dlZ

        if grad:
            return loglike, self._wrap_gradient(dloglike)
        else:
            return loglike

    def sample(self, X, size=None, latent=True, rng=None):
        mu, Sigma = self._predict(X, joint=True)
        rng = rstate(rng)

        # the covariance here is without noise, so the cholesky code may add to
        # the diagonal and raise a warning. since we know this may happen, we
        # can just ignore this.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            L = la.cholesky(Sigma)

        m = 1 if (size is None) else size
        n = len(X)
        f = mu[None] + np.dot(rng.normal(size=(m, n)), L.T)

        if latent is False:
            f = self._like.sample(f.ravel(), rng).reshape(f.shape)
        if size is None:
            f = f.ravel()
        return f

    def predict(self, X, grad=False):
        """
        Return marginal predictions for inputs `X`. Note that the exact form
        that these predictions take will depend on the likelihood model used.
        For example, `Gaussian` likelihoods will return a tuple `(mu, s2)`
        containing the mean and variance for each input; under a `Probit`
        likelihood a vector `p` will be returned which specified the
        probability of observing class 1 for each input.
        """
        return self._like.predict(*self._predict(X, grad))

    def get_tail(self, f, X, grad=False):
        # pylint: disable=arguments-differ
        """
        Compute the expected improvement in value at inputs `X` over the target
        value `f`. If `grad` is true, also compute gradients with respect to
        the inputs.
        """
        return self._like.get_tail(f, *self._predict(X, grad=grad))

    def get_improvement(self, f, X, grad=False):
        # pylint: disable=arguments-differ
        """
        Compute the expected improvement in value at inputs `X` over the target
        value `f`. If `grad` is true, also compute gradients with respect to
        the inputs.
        """
        return self._like.get_improvement(f, *self._predict(X, grad=grad))

    def get_entropy(self, X, grad=False):
        # pylint: disable=arguments-differ
        """
        Compute the predictive entropy evaluated at inputs `X`. If `grad` is
        true, also compute gradients quantity with respect to the inputs.
        """
        return self._like.get_entropy(*self._predict(X, grad=grad))

    def sample_f(self, n, rng=None):
        """
        Return a function or object `f` implementing `__call__` which can be
        used as a sample of the latent function. The argument `n` specifies the
        number of approximate features to use.
        """
        return FourierSample(self._like, self._kern, self._mean,
                             self._X, self._Y, n, rng)

    def bootsample_f(self, n, rng=None):
        return BootStrapSample(self, n, rng)

    def condition_xstar(self, xstar):
        return GP_xstar(self._like, self._kern, self._mean,
                        self._X, self._Y, xstar)

    def condition_fstar(self, fstar):
        return GP_fstar(self._like, self._kern, self._mean,
                        self._X, self._Y, fstar)


def make_gp(sn2, rho, ell, mean=0.0, p=None, kernel='se', inf='exact', U=None):
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

    return GP(like, kern, mean, inf, U)
