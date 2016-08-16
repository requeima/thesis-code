"""
Approximate finite-dimensional samples from a GP.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from ...utils import linalg as la
from ...utils.misc import rstate


class BootStrapSample(object):
    def __init__(self, gp, n, rng=None):
        # resolution = n
        resolution = 100
        # sample from uniform distribution over the bounds
        self.gp = gp.copy()
        self.rng = rng

        ndim = self.gp.ndim
        X = self.gp._X
        Y = self.gp._Y

        sample_X = np.empty((resolution, ndim))


        for d in range(ndim):
            if X is not None:
                a = np.min(X[:, d])
                b = np.max(X[:, d])
                sample_X[:, d] = np.random.uniform(low=a, high=b, size=(resolution))
        sample_Y = np.transpose(self.gp.sample(sample_X, 1, self.rng))

        if Y is not None: # there is an attribute post for this
            self.gp.add_data(np.vstack((X, sample_X)),
                             np.concatenate((Y.flatten(), sample_Y.flatten())) )
        else:
            self.gp.add_data(sample_X,  sample_Y.flatten())


    def __call__(self, x, grad=False):
        if grad:
            # F, G = self.get(np.array(x), True)
            # return F[0][0], G[1][0]
            raise NotImplementedError
        else:
            return self.get(x)[0]

    def get(self, X, grad=False):
        if grad:
            raise NotImplementedError
            # return self.gp.predict(X, grad)[0],
        else:
            # this is really hack but it's what I have to do to get it to work
            try:
                return self.gp.predict(X[None,:])[0]
            except:
                return self.gp.predict(X)[0]
