import numpy as np
import numba as nb
import scipy as sp
from numpy import newaxis as na, linalg as la
import pandas as pd
import sys

from abc import ABCMeta, abstractmethod

"""
Preliminary implementation of routines computing various performance metrics used in state estimation.

Every function expects data in a numpy array of shape (D, N, M, ...), where
D - dimension, N - time steps, M - MC simulations, ... - other optional irrelevant dimensions.
"""


def squared_error(x, m):
    """
    Squared Error

    .. math::
        \mathrm{SE} = (x_k - m_k)^2

    Parameters
    ----------
    x: (dim_x, num_time_steps, num_mc_sims) ndarray
        True state.

    m: (dim_x, num_time_steps, num_mc_sims, num_algs) ndarray
        State mean (state estimate).

    Returns
    -------
    (d, time_steps, mc_sims) ndarray
        Difference between the true state and its estimate squared.
    """
    return (x - m) ** 2


def mse_matrix(x, m):
    """
    Sample Mean Square Error matrix

    Parameters
    ----------
    x: (dim_x, 1) ndarray
        True state.

    m: (dim_x, num_mc_sims) ndarray
        State mean (state estimate).

    Returns
    -------
    : (dim_x, dim_x) ndarray
        Sample mean square error matrix.
    """

    d, mc_sims = m.shape
    dx = x - m
    mse = np.empty((d, d, mc_sims))
    for s in range(mc_sims):
        mse[..., s] = np.outer(dx[..., s], dx[..., s])
    return mse.mean(axis=2)  # average over MC simulations


def log_cred_ratio(x, m, P, MSE):
    """
    Logarithm of Credibility Ratio [Li2006]_ is given by

    .. math::
        \\gamma_n = 10*\\log_10 \\frac{(x - m_n)^{\\top}P_n^{-1}(x - m_n)}{(x - m_n)^{\\top}\Sigma^{-1}(x - m_n)}

    Parameters
    ----------
    x : (dim_x, ) ndarray
        True state.

    m : (dim_x, ) ndarray
        State mean.

    P : (dim_x, dim_x) ndarray
        State covariance matrix.

    MSE : (dim_x, dim_x) ndarray
        Mean square error matrix.

    Returns
    -------
    : float
        Logarithm of credibility ratio.

    Notes
    -----
    Log credibility ratio is defined in [Li2006]_ and is an essential quantity for computing the inclination indicator

    .. math::
        I^2 = \\frac{1}{N}\\sum\\limits_{n=1}^{N} \\gamma_n,

    and the non-credibility index given by

    .. math::
        NCI = \\frac{1}{N}\\sum\\limits_{n=1}^{N} \\| \\gamma_n \\|.

    Since in state estimation examples one can either average over time or MC simulations, the implementation of I^2
    and NCI is left for the user.

    References
    ----------
    .. [Li2006] X. R. Li and Z. Zhao, “Measuring Estimator’s Credibility: Noncredibility Index,”
           in Information Fusion, 2006 9th International Conference on, 2006, pp. 1–8.
    """
    dx = x - m
    sqrtP = mat_sqrt(P)
    sqrtMSE = mat_sqrt(MSE)
    sqrtP_dx = sp.linalg.solve(sqrtP, dx)
    sqrtMSE_dx = sp.linalg.solve(sqrtMSE, dx)
    dx_icov_dx = sqrtP_dx.T.dot(sqrtP_dx)
    dx_imse_dx = sqrtMSE_dx.T.dot(sqrtMSE_dx)
    return 10 * (sp.log10(dx_icov_dx) - sp.log10(dx_imse_dx))


def neg_log_likelihood(x, m, P):
    """
    Negative log-likelihood of the state estimate given the true state.

    Parameters
    ----------
    x : (dim_x, ) ndarray
        True state.

    m : (dim_x, ) ndarray
        State mean.

    P : (dim_x, dim_x) ndarray
        State covariance matrix.

    Returns
    -------
    : float
        Negative logarithm of likelihood of the state given the true state.
    """

    dx = x - m
    d = x.shape[0]
    dx_iP_dx = dx.dot(np.linalg.inv(P)).dot(dx)
    sign, logdet = np.linalg.slogdet(P)
    return 0.5 * (sign*logdet + dx_iP_dx + d * np.log(2 * np.pi))


def kl_divergence(mean_0, cov_0, mean_1, cov_1):
    """
    KL-divergence between the true and approximate Gaussian probability density functions.

    Parameters
    ----------
    mean_0 : (dim_x, ) ndarray
        Mean of the true distribution.

    cov_0 : (dim_x, dim_x) ndarray
        Covariance of the true distribution.

    mean_1 : (dim_x, ) ndarray
        Mean of the approximate distribution.

    cov_1 : (dim_x, dim_x) ndarray
        Covariance of the approximate distribution.

    Returns
    -------
    : float
        KL-divergence of two Gaussian densities.
    """
    k = 1 if np.isscalar(mean_0) else mean_0.shape[0]
    cov_0, cov_1 = np.atleast_2d(cov_0, cov_1)
    dmu = mean_0 - mean_1
    dmu = np.asarray(dmu)
    det_0 = np.linalg.det(cov_0)
    det_1 = np.linalg.det(cov_1)
    inv_1 = np.linalg.inv(cov_1)
    kl = 0.5 * (np.trace(np.dot(inv_1, cov_0)) + np.dot(dmu.T, inv_1).dot(dmu) + np.log(det_0 / det_1) - k)
    return np.asscalar(kl)


def symmetrized_kl_divergence(mean_0, cov_0, mean_1, cov_1):
    """
    Symmetrized KL-divergence

    .. math::
        \\mathrm{SKL} = \\frac{1}{2}[KL(q(x)||p(x)) + KL(p(x)||q(x))]

    between the true :math:`p(x) = \\mathrm{N}(x | m_0, C_0)` and the approximate Gaussian probability density function
     :math:`q(x) = \\mathrm{N}(x | m_1, C_1)`.

    Parameters
    ----------
    mean_0 : (dim_x, ) ndarray
        Mean of the true distribution.

    cov_0 : (dim_x, dim_x) ndarray
        Covariance of the true distribution.

    mean_1 : (dim_x, ) ndarray
        Mean of the approximate distribution.

    cov_1 : (dim_x, dim_x) ndarray
        Covariance of the approximate distribution.

    Returns
    -------
    : float
        Symmetrized KL-divergence of two Gaussian densities.

    Notes
    -----
    Other symmetrizations exist.
    """
    return 0.5 * (kl_divergence(mean_0, cov_0, mean_1, cov_1) + kl_divergence(mean_1, cov_1, mean_0, cov_0))


def bootstrap_var(data, samples=1000):
    """
    Estimates variance of a given data sample by bootstrapping.

    Parameters
    ----------
    data: (1, mc_sims) ndarray
        Data set.
    samples: int, optional
        Number of samples to use during bootstrapping.

    Returns
    -------
    : float
        Bootstrap estimate of variance of the data set.
    """
    data = data.squeeze()
    mc_sims = data.shape[0]
    # sample with replacement to create new datasets
    smp_data = np.random.choice(data, (samples, mc_sims))
    # calculate sample mean of each dataset and variance of the means
    return np.var(np.mean(smp_data, 1))


def print_table(data, row_labels=None, col_labels=None, latex=False):
    pd.DataFrame(data, index=row_labels, columns=col_labels)
    print(pd)
    if latex:
        pd.to_latex()


def gauss_mixture(means, covs, alphas, size):
    """
    Draw samples from Gaussian mixture.

    Parameters
    ----------
    means : tuple of ndarrays
        Mean for each of the mixture components.

    covs : tuple of ndarrays
        Covariance for each of the mixture components.

    alphas : 1d ndarray
        Mixing proportions, must have same length as means and covs.

    size : int or tuple of ints  #TODO: tuple of ints not yet handled.
        Number of samples to draw or shape of the output array containing samples.

    Returns
    -------
    samples : ndarray
        Samples from the Gaussian mixture.

    indexes : ndarray
        Component of indices corresponding to samples in
    """
    if len(means) != len(covs) or len(covs) != len(alphas):
        raise ValueError('means, covs and alphas need to have the same length.')

    n_samples = np.prod(size)
    n_dim = len(means[0])
    # draw from discrete distribution according to the mixing proportions
    ci = np.random.choice(np.arange(len(alphas)), p=alphas, size=size)
    ci_counts = np.unique(ci, return_counts=True)[1]

    # draw samples from each of the component Gaussians
    samples = np.empty((n_samples, n_dim))
    indexes = np.empty(n_samples, dtype=int)
    start = 0
    for ind, c in enumerate(ci_counts):
        end = start + c
        samples[start:end, :] = np.random.multivariate_normal(means[ind], covs[ind], size=c)
        indexes[start:end] = ind
        start = end
    from sklearn.utils import shuffle
    return shuffle(samples, indexes)


def bigauss_mixture(m0, c0, m1, c1, alpha, size):
    """
    Samples from a Gaussian mixture with two components.

    Draw samples of a random variable :math:`X` following a Gaussian mixture density with two components,
    given by

    .. math::
        X \\sim \\alpha \\mathrm{N}(m_0, C_0) + (1 - \\alpha)\\mathrm{N}(m_1, C_1)

    Parameters
    ----------
    m0 : (dim_x, ) ndarray
        Mean of the first component.

    c0 : (dim_x, dim_x) ndarray
        Covariance of the first component.

    m1 : (dim_x, ) ndarray
        Mean of the second component.

    c1 : (dim_x, dim_x) ndarray
        Covariance of the second component.

    alpha : float
        Mixing proportions, alpha.

    size : int or tuple of ints
        Number of samples to draw, gets passed into Numpy's random number generators.

    Returns
    -------
    : ndarray
        Samples of a Gaussian mixture with two components.

    Notes
    -----
    Very inefficient implementation, because it throws away a lot of the samples!
    """
    mi = np.random.binomial(1, alpha, size).T  # 1 w.p. alpha, 0 w.p. 1-alpha
    n0 = np.random.multivariate_normal(m0, c0, size).T
    n1 = np.random.multivariate_normal(m1, c1, size).T
    m1 = (mi[na, ...] == True)
    m0 = np.logical_not(m1)
    return m1 * n0 + m0 * n1


def multivariate_t(mean, scale, nu, size):
    """
    Samples from a multivariate Student's t-distribution.

    Samples of a random variable :math:`X` following a multivariate t-distribution
    :math:`X \\sim \\mathrm{St}(\\mu, \\Sigma, \\nu)`.

    Parameters
    ----------
    mean : (dim_x, ) ndarray
        Mean vector.

    scale : (dim_x, dim_x) ndarray
        Scale matrix.

    nu : float
        Degrees of freedom.

    size : int or tuple of ints
        Number of samples to draw, gets passed into Numpy's random number generators.

    Returns
    -------
    : ndarray
        Samples of a multivariate Student's t-distribution with two components.

    Notes
    -----
    If :math:`y \\sim \\mathrm{N}(0, \\Sigma)` and :math:`u \\sim \\mathrm{Gamma}(k=\\nu/2, \\theta=2/\\nu)`,
    then :math:`x \\sim \\mathrm{St}(\\mu, \\Sigma, \\nu)`, where :math:`x = \\mu + \\frac{y}{\\sqrt{u}}`.
    """
    v = np.random.gamma(nu / 2, 2 / nu, size)[:, na]
    n = np.random.multivariate_normal(np.zeros_like(mean), scale, size)
    return mean[na, :] + n / np.sqrt(v)


def maha(x, y, V=None):
    """
    Mahalanobis distance of all pairs of supplied data points.

    Parameters
    ----------
    x : (num_points, dim_x) ndarray
        Data points.

    y : (num_points, dim_x) ndarray
        Data points.

    V : ndarray (dim_x, dim_x)
        Weight matrix, if `V=None`, `V=eye(D)` is used.

    Returns
    -------
    : (num_points, num_points) ndarray
        Pair-wise Mahalanobis distance of rows of x and y with given weight matrix V.
    """
    if V is None:
        V = np.eye(x.shape[1])
    x2V = np.sum(x.dot(V) * x, 1)
    y2V = np.sum(y.dot(V) * y, 1)
    return (x2V[:, na] + y2V[:, na].T) - 2 * x.dot(V).dot(y.T)


def mat_sqrt(a):
    """
    Matrix square-root.

    Parameters
    ----------
    a : (n, n) ndarray
        Matrix to factor.

    Returns
    -------
    : (n, n) ndarray
        If `a` is symmetric positive-definite, `cholesky(a)` is returned. Otherwise `u.dot(sqrt(s))` is returned,
        where `u, s, v = svd(a)`.
    """
    try:
        b = sp.linalg.cholesky(a, lower=True)
    except np.linalg.linalg.LinAlgError:
        print('Cholesky failed, using SVD.', file=sys.stderr)
        u, s, v = sp.linalg.svd(a)
        b = u.dot(np.diag(np.sqrt(s)))
    return b


def ellipse_points(pos, mat):
    """
    Points on an ellipse given by center position and symmetric positive-definite matrix.

    Parameters
    ----------
    pos : (dim_x) ndarray
          specifying center of the ellipse.

    mat : (dim_x, dim_x) ndarray
          Symmetric positive-definite matrix.

    Returns
    -------
    x : (dim_x, 1) ndarray
        Points on an ellipse defined my the input mean and covariance.
    """
    w, v = la.eig(mat)
    theta = np.linspace(0, 2 * np.pi)
    t = np.asarray((np.cos(theta), np.sin(theta)))
    return pos[:, na] + np.dot(v, np.sqrt(w[:, na]) * t)


def n_sum_k(n, k):
    """Generates all n-tuples summing to k."""
    assert k >= 0
    if k == 0:
        return np.zeros((n, 1), dtype=np.int)
    if k == 1:
        return np.eye(n, dtype=np.int)
    else:
        a = n_sum_k(n, k - 1)
        I = np.eye(n, dtype=np.int)
        temp = np.zeros((n, (n * (1 + n) // 2) - 1), dtype=np.int)
        tind = 0
        for i in range(n - 1):
            for j in range(i, n):
                temp[:, tind] = a[:, i] + I[:, j]
                tind = tind + 1
        return np.hstack((temp, a[:, n - 1:] + I[:, -1, None]))


@nb.jit(nopython=True)
def vandermonde(mul_ind, x):
    """
    Vandermonde matrix with multivariate polynomial basis.

    Parameters
    ----------
    mul_ind : (dim, num_basis) ndarray
        Matrix where each column is a multi-index which specifies a multivariate monomial.

    x : (dim, num_points) ndarray
        Sigma-points.

    Returns
    -------
    : (num_points, num_basis) ndarray
        Vandermonde matrix evaluated for all sigma-points.
    """
    dim, num_pts = x.shape
    num_basis = mul_ind.shape[1]
    vdm = np.zeros((num_pts, num_basis))
    for n in range(num_pts):
        for b in range(num_basis):
            vdm[n, b] = np.prod(x[:, n] ** mul_ind[:, b])
    return vdm


class RandomVariable(metaclass=ABCMeta):

    @abstractmethod
    def sample(self, size):
        pass

    @abstractmethod
    def get_stats(self):
        pass


class GaussRV(RandomVariable):

    def __init__(self, dim, mean=None, cov=None):
        self.dim = dim

        # standard Gaussian distribution if mean, cov not specified
        if mean is None:
            mean = np.zeros((dim, ))
        if mean.ndim != 1:
            ValueError(
                "{:s}: mean has to be 1D array. Supplied {:d}D array.".format(self.__class__.__name__, mean.ndim))
        self.mean = mean

        if cov is None:
            cov = np.eye(dim)
        if cov.ndim != 2:
            ValueError(
                "{:s}: covariance has to be 2D array. Supplied {:d}D array.".format(self.__class__.__name__, cov.ndim))
        self.cov = cov

    def sample(self, size):
        return np.random.multivariate_normal(self.mean, self.cov, size)

    def get_stats(self):
        return self.mean, self.cov