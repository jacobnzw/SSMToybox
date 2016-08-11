import numpy as np
from numpy import newaxis as na
import pandas as pd

"""
Preliminary implementation of routines computing various performance metrics used in state estimation.

Every function expects data in a numpy array of shape (D, N, M, ...), where
D - dimension, N - time steps, M - MC simulations, ... - other optional irrelevant dimensions.
"""


def rmse(x, m):
    """
    Root Mean Squared Error

    .. math::

    \sqrt{\frac{1}{K}\sum\limits_{k=1}^{K}(x_k - m_k)^{\top}(x_k - m_k)}

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

    dx = x[..., na] - m
    MSE = (dx[:, 1:, ...] ** 2).mean(axis=1)  # average over time steps
    return np.sqrt(MSE)


def nci(x, m, P):
    """
    Non-credibility index of the state estimate

    .. math::

        \frac{10}{K}\sum\limits_{k=1}^{K} \log_10 \frac{(x - m)^{\top}P^{-1}(x - m)}{(x - m)^{\top}\Sig^{-1}(x - m)},

    where the :math:`\Sig` is the MSE matrix.

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
    # dimension of state, # time steps, # MC simulations, # inference algorithms (filters/smoothers)
    d, time, mc_sims, algs = m.shape
    dx = x[..., na] - m
    # Mean Square Error matrix
    MSE = np.empty((d, d, time, mc_sims, algs))
    # TODO: these loops can be moved to a general function which calls the specific metric
    for k in range(time):
        for s in range(mc_sims):
            for alg in range(algs):
                MSE[..., k, s, alg] = np.outer(dx[..., k, s, alg], dx[..., k, s, alg])
    MSE = MSE.mean(axis=3)  # average over MC simulations

    # dx_iP_dx = np.empty((1, time, mc_sims, algs))
    NCI = np.empty((1, time, mc_sims, algs))
    for k in range(1, time):
        for s in range(mc_sims):
            for alg in range(algs):
                # iP_dx = cho_solve(cho_factor(P[:, :, k, s, alg]), dx[:, k, s, alg])
                # dx_iP_dx[:, k, s, alg] = dx[:, k, s, alg].dot(iP_dx)
                # iMSE_dx = cho_solve(cho_factor(MSE[..., k, fi]), dx[:, k, s, alg])
                # NCI[..., k, s, fi] = 10*np.log10(dx_iP_dx[:, k, s, fi]) - 10*np.log10(dx[:, k, s, alg].dot(iMSE_dx))
                dx_iP_dx = dx[:, k, s, alg].dot(np.linalg.inv(P[..., k, s, alg])).dot(dx[:, k, s, alg])
                dx_iMSE_dx = dx[:, k, s, alg].dot(np.linalg.inv(MSE[..., k, alg])).dot(dx[:, k, s, alg])
                NCI[..., k, s, alg] = 10 * np.log10(dx_iP_dx) - 10 * np.log10(dx_iMSE_dx)
    return NCI[:, 1:, ...].mean(axis=1)  # average over time steps (ignore the 1st)


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
    d, time, mc_sims, algs = m.shape
    dx = x[..., na] - m
    NLL = np.empty((1, time, mc_sims, algs))
    dx_iP_dx = np.empty((1, time, mc_sims, algs))
    for k in range(1, time):
        for s in range(mc_sims):
            for fi in range(algs):
                S = P[..., k, s, fi]
                dx_iP_dx[:, k, s, fi] = dx[:, k, s, fi].dot(np.linalg.inv(S)).dot(dx[:, k, s, fi])
                NLL[:, k, s, fi] = 0.5 * (np.log(np.linalg.det(S)) + dx_iP_dx[:, k, s, fi] + d * np.log(2 * np.pi))
    return NLL[:, 1:, ...].mean(axis=1)  # average over time steps (ignore the 1st)


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
    #
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
