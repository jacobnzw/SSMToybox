import numpy as np
from numpy import newaxis as na
import pandas as pd

"""
Preliminary implementation of routines computing various performance metrics used in state estimation.

Every function expects data in a numpy array of shape (D, N, M, ...), where
D - dimension, N - time steps, M - MC simulations, ... - other optional irrelevant dimensions.
"""


def squared_error(x, m):
    """
    Squared Error

    .. math::

    SE = (x_k - m_k)^2

    Parameters
    ----------
    x: numpy.ndarray with shape (d, time_steps, mc_sims)
        True state

    m: numpy.ndarray with shape (d, time_steps, mc_sims, algs)
        State mean

    Returns
    -------
    (d, time_steps, mc_sims)

    """

    return (x - m) ** 2


def mse_matrix(x, m):
    """
    Sample Mean Square Error matrix

    Parameters
    ----------
    x
    m

    Returns
    -------

    """

    d, mc_sims = m.shape
    dx = x - m
    mse = np.empty((d, d, mc_sims))
    for s in range(mc_sims):
        mse[..., s] = np.outer(dx[..., s], dx[..., s])
    return mse.mean(axis=2)  # average over MC simulations


def log_cred_ratio(x, m, P, MSE):
    """
    Logarithm of Credibility Ratio [1]_ is given by

    .. math::

    \gamma_n = 10*\log_10 \frac{(x - m_n)^{\top}P_n^{-1}(x - m_n)}{(x - m_n)^{\top}\Sig^{-1}(x - m_n)}

    Parameters
    ----------
    x:
        True state
    m:
        State mean
    P:
        State covariance matrix
    MSE:
        Mean square error matrix

    Returns
    -------
    :float


    Notes
    -----
    Log credibility ratio is defined in [1]_ and is an essential quantity for computing the inclination indicator

    .. math::

    I^2 = \frac{1}{N}\sum\limits_{n=1}^{N} \gamma_n,

    and the non-credibility index given by

    .. math::

    NCI = \frac{1}{N}\sum\limits_{n=1}^{N} \| \gamma_n \|.

    Since in state estimation examples one can either average over time or MC simulations, the implementation of I^2
    and NCI is left for the user.


    References
    ----------
    .. [1] X. R. Li and Z. Zhao, “Measuring Estimator’s Credibility: Noncredibility Index,”
           in Information Fusion, 2006 9th International Conference on, 2006, pp. 1–8.

    """

    dx = x - m
    dx_icov_dx = dx.dot(np.linalg.inv(P)).dot(dx)
    dx_imse_dx = dx.dot(np.linalg.inv(MSE)).dot(dx)
    return 10 * (np.log10(dx_icov_dx) - np.log10(dx_imse_dx))


def nll(x, m, P):
    """
    Negative log-likelihood of the state estimate given the true state.

    Parameters
    ----------
    x:
        True state
    m:
        State mean
    P:
        State covariance

    Returns
    -------

    """

    dx = x - m
    d = x.shape[0]
    dx_iP_dx = dx.dot(np.linalg.inv(P)).dot(dx)
    sign, logdet = np.linalg.slogdet(P)
    return 0.5 * (sign*logdet + dx_iP_dx + d * np.log(2 * np.pi))


def kl(mean_0, cov_0, mean_1, cov_1):
    """
    KL-divergence

    Parameters
    ----------
    mean_0
    cov_0
    mean_1
    cov_1

    Returns
    -------
    :float
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


def skl(mean_0, cov_0, mean_1, cov_1):
    """
    Symmetrized KL-divergence :math:`0.5[KL(q(x)||p(x)) + KL(p(x)||q(x))]`

    Parameters
    ----------
    mean_0
    cov_0
    mean_1
    cov_1

    Returns
    -------
    :float
        Symmetrized KL-divergence
    """
    return 0.5 * (kl(mean_0, cov_0, mean_1, cov_1) + kl(mean_1, cov_1, mean_0, cov_0))


def bootstrap_var(data, samples=1000):
    """
    Estimates variance of a given data sample by bootstrapping

    Parameters
    ----------
    data: numpy.ndarray (1, mc_sims)
        Data set
    samples: int
        how many samples to use during bootstrapping

    Returns
    -------
    : float
        Bootstrap estimate of variance of the data set.
    """

    # data
    data = data.squeeze()
    mc_sims = data.shape[0]
    # sample with replacement to create new datasets
    smp_data = np.random.choice(data, (samples, mc_sims))
    # calculate sample mean of each dataset and variance of the means
    var = np.var(np.mean(smp_data, 1))
    return 2 * np.sqrt(var)  # 2*STD


def print_table(data, row_labels=None, col_labels=None, latex=False):
    pd.DataFrame(data, index=row_labels, columns=col_labels)
    print(pd)
    if latex:
        pd.to_latex()
