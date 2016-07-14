import numpy as np
from numpy import newaxis as na
import pandas as pd

"""
Preliminary implementation of routines computing various performance metrics used in filtering.

Every function expects data in a numpy array of shape (D, N, M, ...), where
D - dimension, N - time steps, M - MC simulations, ... - other optional irrelevant dimensions.
"""


def rmse(x, m):
    """
    Root Mean Squared Error

    Parameters
    ----------
    x numpy.ndarray of (d, time_steps, mc_sims)
        True state
    m numpy.ndarray of (d, time_steps, mc_sims, algs)

    Returns
    -------
    (d, time_steps, mc_sims)

    """

    dx = x[..., na] - m
    MSE = (dx[:, 1:, ...] ** 2).mean(axis=1)  # average over time steps
    return np.sqrt(MSE)


def nci(x, m, P):
    # dimension of state, # time steps, # MC simulations, # inference algorithms (filters/smoothers)
    d, time, mc_sims, algs = m.shape
    dx = x[..., na] - m
    # Mean Square Error matrix
    MSE = np.empty((d, d, time, mc_sims, algs))
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


def bootstrap_var(data, samples=1000):
    # data (1, mc_sims)
    data = data.squeeze()
    mc_sims = data.shape[0]
    # sample with replacement to create new datasets
    smp_data = np.random.choice(data, (samples, mc_sims))
    # calculate sample mean of each dataset and variance of the means
    var = np.var(np.mean(smp_data, 1))
    return 2 * np.sqrt(var)  # 2*STD


def print_table(data, row_labels=None, col_labels=None, latex=False):
    pd.DataFrame(data, index=row_labels, columns=col_labels)
    print pd
    if latex:
        pd.to_latex()
