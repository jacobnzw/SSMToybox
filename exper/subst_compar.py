import numpy as np
from numpy import newaxis as na
from bq.bqkern import RBF
import matplotlib.pyplot as plt

num_samples = int(1e3)
num_mc = 1
dim = 5  # dimension of the integration domain
kernel_par = np.array([[1] + dim*[.1]])
rbf_kernel = RBF(dim, kernel_par)


def weights(par, x):
    """BQ weights given points and kernel parameters."""

    # inverse kernel matrix
    iK = rbf_kernel.eval_inv_dot(par, x, scaling=False)

    # Kernel expectations
    q = rbf_kernel.exp_x_kx(par, x)

    # BQ weights in terms of kernel expectations
    return q.dot(iK)


def sum_of_squares(x):
    return x.T.dot(x)


# assign integrand
g = sum_of_squares

# mean, covariance of input gaussian
m0 = np.zeros((dim, ))
P0 = np.eye(dim)
m = np.ones((dim,))
P = np.random.randn(dim, dim)
P = P.T.dot(P)
L = np.linalg.cholesky(P)

# true value of integral
quad_true = m.T.dot(m) + np.trace(P)

sample_steps = np.logspace(1, np.log10(num_samples), num=5, dtype=int)

num_sample_steps = len(sample_steps)
error_xi = np.zeros((num_sample_steps, num_mc))
error_eta = np.zeros((num_sample_steps, num_mc))
error_x = np.zeros((num_sample_steps, num_mc))

for mc in range(num_mc):

    # standard gaussian samples
    xi = np.random.multivariate_normal(m0, P0, size=(num_samples, )).T
    eta = m[:, na] + xi  # shifted samples
    x = m[:, na] + L.dot(xi)  # affine samples

    # for each sample
    for i, s in enumerate(sample_steps):

        # evaluate integrand
        g_xi = np.apply_along_axis(g, 0, x[:, :s])
        g_eta = np.apply_along_axis(g, 0, x[:, :s])
        g_x = np.apply_along_axis(g, 0, x[:, :s])

        # compute BQ weights
        w_xi = weights(kernel_par, xi[:, :s])
        w_eta = weights(kernel_par, eta[:, :s])
        w_x = weights(kernel_par, x[:, :s])

        # compute BQ approximations
        quad_xi = w_xi.T.dot(g_xi)
        quad_eta = w_eta.T.dot(g_eta)
        quad_x = w_x.T.dot(g_x)

        error_xi[i, mc] = np.abs(quad_true - quad_xi)
        error_eta[i, mc] = np.abs(quad_true - quad_eta)
        error_x[i, mc] = np.abs(quad_true - quad_x)

# plopt = {'ls':}
plt.semilogx(sample_steps, error_xi.mean(axis=1), 'o-', lw=2, label=r'$ \xi $')
plt.semilogx(sample_steps, error_eta.mean(axis=1), '^-', lw=2, label=r'$ \eta $')
plt.semilogx(sample_steps, error_x.mean(axis=1), 'v-', lw=2, label=r'$x$')
plt.legend()
plt.show()
