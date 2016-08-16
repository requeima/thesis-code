"""
Demo showing GP predictions in 1d and optimization of the hyperparameters.
"""

import numpy as np
import scipy.special as ss

import ezplot as ez
import reggie as rg

# create a gp.
like = rg.likelihoods.Probit()
kern = rg.kernels.MaternARD(1, 0.5, d=3)
mean = rg.means.Zero()
gp = rg.GP(like, kern, mean, inf='laplace')

# sample from it
rng = np.random.RandomState()
n = 500
N = 100
x = np.linspace(-2, 2, n)
f = gp.sample(x[:, None], latent=True, rng=rng)

i = rng.randint(n, size=N)
X = x[i][:, None]
Y = like.sample(f[i], rng=rng)

# create a new gp
gp = rg.GP(like, rg.kernels.SEARD(1, 1), mean, inf='laplace')
gp.add_data(X, Y)
gp.optimize()

# get the posterior moments
p = gp.predict(x[:, None])

# plot the posterior
fig = ez.figure(w_pad=3)
ax1 = fig.add_subplot(2, 1, 1, hidexy=True)
ax2 = fig.add_subplot(2, 1, 2, hidexy=True, sharex=ax1)

ax1.plot(x, ss.ndtr(f))
ax1.set_ylim(0, 1)
ax1.set_title('sigma(f) for f sampled from a Matern-3')

Y[Y == -1] = 0
ax2.plot(x, p)
ax2.scatter(X.ravel(), Y)

# draw/show it
fig.canvas.draw()
ez.show()
