"""
Demo which illustrates how to use solve_bayesopt as a simple method for global
optimization. The return values are the sequence of recommendations made by the
algorithm as well as the final model. The point `xbest[-1]` is the final
recommendation, i.e. the expected maximizer.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import ezplot as ez

from pybo import solve_bayesopt
from pybo.domains import Grid

__all__ = []


def f(x):
    """
    Test function that we will optimize. This is a simple sinusoidal function
    whose maximum should be found very quickly.
    """
    x = float(x)
    return -np.cos(x) - np.sin(3*x)


# solve the test function
bounds = [0, 2*np.pi]
domain = Grid(bounds, 500)
xbest, model, info = solve_bayesopt(f, domain, niter=30, verbose=True)

# make some predictions
mu, s2 = model.predict(domain.X)

# plot the final model
ax = ez.figure().gca()
ax.plot_banded(domain.X.ravel(), mu, 2*np.sqrt(s2))
ax.axvline(xbest)
ax.scatter(info.x.ravel(), info.y)
ax.figure.canvas.draw()
ez.show()
