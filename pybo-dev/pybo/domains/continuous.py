"""
Implementation of methods for sampling initial points.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.optimize as so

from ..utils import rstate, register
from ._sobol import i4_sobol_generate

__all__ = ['init_middle', 'init_uniform', 'init_latin', 'init_sobol',
           'solve_lbfgs']

INITS = {}
SOLVERS = {}


# INITIALIZATION METHODS ######################################################

@register(INITS, 'middle')
def init_middle(bounds, rng=None):
    """
    Initialize using a single query in the middle of the space.
    """
    return np.mean(bounds, axis=1)[None, :]


@register(INITS, 'uniform')
def init_uniform(bounds, n=None, rng=None):
    """
    Initialize using `n` uniformly distributed query points. If `n` is `None`
    then use 3D points where D is the dimensionality of the input space.
    """
    rng = rstate(rng)
    bounds = np.array(bounds, ndmin=2, copy=False)
    d = len(bounds)
    n = 3*d if (n is None) else n

    # generate the random values.
    w = bounds[:, 1] - bounds[:, 0]
    X = bounds[:, 0] + w * rng.rand(n, d)

    return X


@register(INITS, 'latin')
def init_latin(bounds, n=None, rng=None):
    """
    Initialize using a Latin hypercube design of size `n`. If `n` is `None`
    then use 3D points where D is the dimensionality of the input space.
    """
    rng = rstate(rng)
    bounds = np.array(bounds, ndmin=2, copy=False)
    d = len(bounds)
    n = 3*d if (n is None) else n

    # generate the random samples.
    w = bounds[:, 1] - bounds[:, 0]
    X = bounds[:, 0] + w * (np.arange(n)[:, None] + rng.rand(n, d)) / n

    # shuffle each dimension.
    for i in xrange(d):
        X[:, i] = rng.permutation(X[:, i])

    return X


@register(INITS, 'sobol')
def init_sobol(bounds, n=None, rng=None):
    """
    Initialize using a Sobol sequence of length `n`. If `n` is `None` then use
    3D points where D is the dimensionality of the input space.
    """
    rng = rstate(rng)
    bounds = np.array(bounds, ndmin=2, copy=False)
    d = len(bounds)
    n = 3*len(bounds) if (n is None) else n

    # generate the random samples.
    skip = rng.randint(100, 200)
    w = bounds[:, 1] - bounds[:, 0]
    X = bounds[:, 0] + w * i4_sobol_generate(d, n, skip).T

    return X


# SOLVER METHODS ##############################################################

@register(SOLVERS, 'lbfgs')
def solve_lbfgs(f,
                bounds,
                nbest=10,
                ngrid=10000,
                xgrid=None,
                rng=None):
    """
    Compute the objective function on an initial grid, pick `nbest` points, and
    maximize using LBFGS from these initial points.

    Args:
        f: function handle that takes an optional `grad` boolean kwarg
           and if `grad=True` returns a tuple of `(function, gradient)`.
           NOTE: this functions is assumed to allow for multiple inputs in
           vectorized form.

        bounds: bounds of the search space.
        nbest: number of best points from the initial test points to refine.
        ngrid: number of (random) grid points to test initially.
        xgrid: initial test points; ngrid is ignored if this is given.

    Returns:
        xmin, fmax: location and value of the maximizer.
    """

    if xgrid is None:
        # TODO: The following line could be replaced with a regular grid or a
        # Sobol grid.
        xgrid = init_uniform(bounds, ngrid, rng)
    else:
        xgrid = np.array(xgrid, ndmin=2)

    # compute func_grad on points xgrid
    finit = f(xgrid, grad=False)
    idx_sorted = np.argsort(finit)[::-1]

    # lbfgsb needs the gradient to be "contiguous", squeezing the gradient
    # protects against func_grads that return ndmin=2 arrays. We also need to
    # negate everything so that we are maximizing.
    def objective(x):
        fx, gx = f(x[None], grad=True)
        return -fx[0], -gx[0]

    # TODO: the following can easily be multiprocessed
    result = [so.fmin_l_bfgs_b(objective, x0, bounds=bounds)[:2]
              for x0 in xgrid[idx_sorted[:nbest]]]

    # loop through the results and pick out the smallest.
    xmin, fmin = result[np.argmin(_[1] for _ in result)]

    # return the values (negate if we're finding a max)
    return xmin, -fmin


# DIRECT ######################################################################

try:
    # This will try and import the nlopt package in order to use the DIRECT
    # (DIvided RECTangles) solver for gradient-free optimization. if nlopt
    # doesn't exist we'll just ignore it.
    import nlopt

    # exported symbols
    __all__ += ['solve_direct']

    @register(SOLVERS, 'direct')
    def solve_direct(f, bounds, rng=None):
        def objective(x, grad):
            """Objective function in the form required by nlopt."""
            if grad.size > 0:
                fx, gx = f(x[None], grad=True)
                grad[:] = gx[0][:]
            else:
                fx = f(x[None], grad=False)
            return fx[0]

        bounds = np.array(bounds, ndmin=2)

        opt = nlopt.opt(nlopt.GN_ORIG_DIRECT, bounds.shape[0])
        opt.set_lower_bounds(list(bounds[:, 0]))
        opt.set_upper_bounds(list(bounds[:, 1]))
        opt.set_ftol_rel(1e-6)
        opt.set_maxtime(10)
        opt.set_max_objective(objective)

        xmin = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) / 2
        xmin = opt.optimize(xmin)
        fmax = opt.last_optimum_value()

        return xmin, fmax

except ImportError:
    pass
