"""
Inference for GP regression.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from .gp import GP
from .conditional import GP_xstar_light


__all__ = ['GPLight']


class GPLight(GP):
    def __init__(self, like, kern, mean, inf='exact', U=None):
        super(GPLight, self).__init__(like, kern, mean, inf, U)

    def condition_xstar(self, xstar):
        return GP_xstar_light(self._like, self._kern, self._mean,
                        self._X, self._Y, xstar)