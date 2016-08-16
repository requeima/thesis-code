"""
Definition of the function interface.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ..core.params import Parameterized

__all__ = ['Mean']


# BASE KERNEL INTERFACE #######################################################

class Mean(Parameterized):
    """
    The base interface for mean functions.
    """
    def __call__(self, x):
        X = np.array(x, ndmin=1)[None]
        return self.get_mean(X)[0]

    def get_mean(self, X):
        """
        Evaluate the function at input points X.
        """
        raise NotImplementedError

    def get_grad(self, X):
        """
        Get the gradient of the function with respect to any hyperparameters,
        evaluated at input points X. Return a generator yielding each gradient
        component.
        """
        raise NotImplementedError


class RealMean(Mean):
    """
    Mean function defined over a real-valued input space.
    """

    def get_gradx(self, X):
        """
        Return the gradient with respect to the input space.
        """
        raise NotImplementedError
