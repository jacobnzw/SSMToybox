import numpy as np
from numpy import newaxis as na
# import matplotlib.pyplot as plt
from fusion_paper.figprint import *
from transforms.bqmodel import GaussianProcess, StudentTProcess
from utils import multivariate_t
from transforms.bqkernel import RBF

dim = 1
par_kernel = np.array([[0.8, 0.7]])

# init models
gp = GaussianProcess(dim, par_kernel)
tp = StudentTProcess(dim, par_kernel, nu=10.0)

# some nonlinear function
f = lambda x: np.sin(np.sin(x)*x**2)*np.exp(x)
expit = lambda x: 5/(1+np.exp(-20*(x+1)))+0.01 + 5/(1+np.exp(20*(x-2)))+0.01

# setup some test data
num_test = 100
x_test = np.linspace(-5, 5, num_test)[na, :]

# draw from a GP
K = gp.kernel.eval(np.array([[0.1, 0.7]]), x_test) + 1e-8*np.eye(num_test)
gp_sample = np.random.multivariate_normal(np.zeros(num_test), K)
# amplitude modulation of the gp sample
gp_sample *= expit(np.linspace(-5, 5, num_test))
gp_sample += expit(np.linspace(-5, 5, num_test))


i_train = [10, 20, 40, 52, 55, 80]
x_train = x_test[:, i_train]
y_train = gp_sample[i_train]  # + multivariate_t(np.zeros((1,)), 0.5*np.eye(1), 3.0, size=len(i_train)).squeeze()
# noise = multivariate_t(np.zeros((1,)), np.eye(1), 3.0, size=gp.num_pts).T
y_test = gp_sample

gp_mean, gp_var = gp.predict(x_test, y_train, x_train, par_kernel)
gp_std = np.sqrt(gp_var)
tp_mean, tp_var = tp.predict(x_test, y_train, x_train, par_kernel)
tp_std = np.sqrt(tp_var)

x_test = x_test.squeeze()
y_test = y_test.squeeze()
x_train = x_train.squeeze()
y_train = y_train.squeeze()

fp = FigurePrint()

# plt.plot(np.linspace(-5, 5, num_test), expit(np.linspace(-5, 5, num_test)))
# plot training data, predictive mean and variance
ymin, ymax, ypad = gp_sample.min(), gp_sample.max(), 0.25*gp_sample.ptp()
fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].fill_between(x_test, gp_mean - 2 * gp_std, gp_mean + 2 * gp_std, color='0.1', alpha=0.15)
ax[0].plot(x_test, gp_mean, color='k', lw=2)
ax[0].plot(x_train, y_train, 'ko', ms=6)
ax[0].plot(x_test, y_test, lw=2, ls='--', color='tomato')
ax[0].set_ylim([ymin-ypad, ymax+ypad])
ax[0].set_ylabel('g(x)')

ax[1].fill_between(x_test, tp_mean - 2 * tp_std, tp_mean + 2 * tp_std, color='0.1', alpha=0.15)
ax[1].plot(x_test, tp_mean, color='k', lw=2)
ax[1].plot(x_train, y_train, 'ko', ms=6)
ax[1].plot(x_test, y_test, lw=2, ls='--', color='tomato')
ax[1].set_ylim([ymin-ypad, ymax+ypad])
ax[1].set_ylabel('g(x)')
ax[1].set_xlabel('x')

plt.tight_layout(pad=0.0)
plt.show()
fp.savefig('gp_vs_tp')