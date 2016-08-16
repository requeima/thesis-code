"""
Animated demo showing optimization of a bandit with independent arms.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import ezplot as ez
import reggie as rg


# define the model and sample an instance from it
n = 10
rng = np.random.RandomState()
model = rg.BetaBernoulli(np.ones(n))
f = model.sample(rng=rng)

# grab the optimal latent value and make lists of observations
xopt = f.argmax()
fopt = f.max()

# create a new figure
fig = ez.figure(figsize=(10, 6))

while True:
    # get our index
    mu, lo, hi = (model.get_quantile(q) for q in (0.5, 0.05, 0.95))
    target = mu.max()
    index = model.get_improvement(target)

    # query
    x = index.argmax()
    y = int(rng.uniform() < f[x])

    # add the data
    model.add_data(x, y)

    # PLOT EVERYTHING
    fig.clear()
    ax1 = fig.add_subplotspec((2, 1), (0, 0), hidex=True)
    ax2 = fig.add_subplotspec((2, 1), (1, 0), hidey=True, sharex=ax1)

    ax1.errorbar(np.arange(n), mu, (mu-lo, hi-mu), ls='', marker='s',
                 markersize=20, capsize=30, capthick=2)
    ax1.axvline(xopt, zorder=-1)
    ax1.axhline(fopt, zorder=-1)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(-0.3, n-1+0.3)

    ax2.bar(np.arange(n)-0.25, index, 0.5)

    # draw
    fig.canvas.draw()
    ez.show(block=False)
