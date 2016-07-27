import numpy as np
import pandas as pd
from numpy import newaxis as na
from scipy.stats import multivariate_normal
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import cholesky
from inference import ExtendedKalman, CubatureKalman, UnscentedKalman, GaussHermiteKalman, GPQuadKalman
from transforms import SphericalRadial, Unscented, GaussHermite, GPQuad
from ungm import UNGM
import matplotlib.pyplot as plt
import time


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
    print(pd)
    if latex:
        pd.to_latex()


def tables():
    steps, mc = 500, 100
    ssm = UNGM()  # initialize UNGM model
    x, z = ssm.simulate(steps, mc_sims=mc)  # generate some data
    # initialize filters/smoothers
    algorithms = (
        # ExtendedKalman(ssm),
        CubatureKalman(ssm),
        UnscentedKalman(ssm, kappa=0.0),
        GaussHermiteKalman(ssm, deg=5),
        GaussHermiteKalman(ssm, deg=7),
        GaussHermiteKalman(ssm, deg=10),
        GaussHermiteKalman(ssm, deg=15),
        GaussHermiteKalman(ssm, deg=20),
        GPQuadKalman(ssm,
                     usp_dyn=SphericalRadial.unit_sigma_points(ssm.xD),
                     usp_meas=SphericalRadial.unit_sigma_points(ssm.xD),
                     hyp_dyn={'sig_var': 1.0, 'lengthscale': 0.3 * np.ones(ssm.xD, ), 'noise_var': 1e-8},
                     hyp_meas={'sig_var': 1.0, 'lengthscale': 0.3 * np.ones(ssm.xD, ), 'noise_var': 1e-8}),
        GPQuadKalman(ssm,
                     usp_dyn=Unscented.unit_sigma_points(ssm.xD, kappa=0.0),
                     usp_meas=Unscented.unit_sigma_points(ssm.xD, kappa=0.0),
                     hyp_dyn={'sig_var': 1.0, 'lengthscale': 3.0 * np.ones(ssm.xD, ), 'noise_var': 1e-8},
                     hyp_meas={'sig_var': 1.0, 'lengthscale': 3.0 * np.ones(ssm.xD, ), 'noise_var': 1e-8}),
        GPQuadKalman(ssm,
                     usp_dyn=GaussHermite.unit_sigma_points(ssm.xD, degree=5),
                     usp_meas=GaussHermite.unit_sigma_points(ssm.xD, degree=5),
                     hyp_dyn={'sig_var': 1.0, 'lengthscale': 0.3 * np.ones(ssm.xD, ), 'noise_var': 1e-8},
                     hyp_meas={'sig_var': 1.0, 'lengthscale': 0.3 * np.ones(ssm.xD, ), 'noise_var': 1e-8}),
        GPQuadKalman(ssm,
                     usp_dyn=GaussHermite.unit_sigma_points(ssm.xD, degree=7),
                     usp_meas=GaussHermite.unit_sigma_points(ssm.xD, degree=7),
                     hyp_dyn={'sig_var': 1.0, 'lengthscale': 0.1 * np.ones(ssm.xD, ), 'noise_var': 1e-8},
                     hyp_meas={'sig_var': 1.0, 'lengthscale': 0.1 * np.ones(ssm.xD, ), 'noise_var': 1e-8}),
        GPQuadKalman(ssm,
                     usp_dyn=GaussHermite.unit_sigma_points(ssm.xD, degree=10),
                     usp_meas=GaussHermite.unit_sigma_points(ssm.xD, degree=10),
                     hyp_dyn={'sig_var': 1.0, 'lengthscale': 0.1 * np.ones(ssm.xD, ), 'noise_var': 1e-8},
                     hyp_meas={'sig_var': 1.0, 'lengthscale': 0.1 * np.ones(ssm.xD, ), 'noise_var': 1e-8}),
        GPQuadKalman(ssm,
                     usp_dyn=GaussHermite.unit_sigma_points(ssm.xD, degree=15),
                     usp_meas=GaussHermite.unit_sigma_points(ssm.xD, degree=15),
                     hyp_dyn={'sig_var': 1.0, 'lengthscale': 0.1 * np.ones(ssm.xD, ), 'noise_var': 1e-8},
                     hyp_meas={'sig_var': 1.0, 'lengthscale': 0.1 * np.ones(ssm.xD, ), 'noise_var': 1e-8}),
        GPQuadKalman(ssm,
                     usp_dyn=GaussHermite.unit_sigma_points(ssm.xD, degree=20),
                     usp_meas=GaussHermite.unit_sigma_points(ssm.xD, degree=20),
                     hyp_dyn={'sig_var': 1.0, 'lengthscale': 0.1 * np.ones(ssm.xD, ), 'noise_var': 1e-8},
                     hyp_meas={'sig_var': 1.0, 'lengthscale': 0.1 * np.ones(ssm.xD, ), 'noise_var': 1e-8})
    )
    num_algs = len(algorithms)

    # space for estimates
    mean_f, cov_f = np.zeros((ssm.xD, steps, mc, num_algs)), np.zeros((ssm.xD, ssm.xD, steps, mc, num_algs))
    mean_s, cov_s = np.zeros((ssm.xD, steps, mc, num_algs)), np.zeros((ssm.xD, ssm.xD, steps, mc, num_algs))
    # do filtering/smoothing
    t0 = time.time()  # measure execution time
    print('Running filters/smoothers ...')
    for a, alg in enumerate(algorithms):
        print('{}'.format(alg.__class__.__name__))  # print filter/smoother name
        for sim in range(mc):
            mean_f[..., sim, a], cov_f[..., sim, a] = alg.forward_pass(z[..., sim])
            mean_s[..., sim, a], cov_s[..., sim, a] = alg.backward_pass()
            alg.reset()
    print('Done in {0:.4f} [sec]'.format(time.time() - t0))

    # evaluate perfomance
    rmseData_f, rmseData_s = rmse(x, mean_f), rmse(x, mean_s)  # averaged RMSE over time steps
    nciData_f, nciData_s = nci(x, mean_f, cov_f), nci(x, mean_s, cov_s)  # averaged NCI over time steps
    nllData_f, nllData_s = nll(x, mean_f, cov_f), nll(x, mean_s, cov_s)  # averaged NCI over time steps
    # average scores (over MC simulations)
    rmseMean_f, rmseMean_s = rmseData_f.mean(axis=1).T, rmseData_s.mean(axis=1).T
    nciMean_f, nciMean_s = nciData_f.mean(axis=1).T, nciData_s.mean(axis=1).T
    nllMean_f, nllMean_s = nllData_f.mean(axis=1).T, nllData_s.mean(axis=1).T
    # +/- 2 standard deviations of the scores (using bootstrapping)
    rmseStd_f, rmseStd_s = np.zeros(num_algs), np.zeros(num_algs)
    nciStd_f, nciStd_s = rmseStd_f.copy(), rmseStd_f.copy()
    nllStd_f, nllStd_s = rmseStd_f.copy(), rmseStd_f.copy()
    for f in range(num_algs):
        rmseStd_f[f] = bootstrap_var(rmseData_f[..., f], samples=1e4)
        rmseStd_s[f] = bootstrap_var(rmseData_s[..., f], samples=1e4)
        nciStd_f[f] = bootstrap_var(nciData_f[..., f], samples=1e4)
        nciStd_s[f] = bootstrap_var(nciData_s[..., f], samples=1e4)
        nllStd_f[f] = bootstrap_var(nllData_f[..., f], samples=1e4)
        nllStd_s[f] = bootstrap_var(nllData_s[..., f], samples=1e4)

    # put data into Pandas DataFrame for fancy printing and latex export
    row_labels = ['SR', 'UT', 'GH-5', 'GH-7', 'GH-10', 'GH-15',
                  'GH-20']  # [alg.__class__.__name__ for alg in algorithms]
    col_labels = ['Classical', 'Bayesian', 'Classical (2std)', 'Bayesian (2std)']
    rmse_table_f = pd.DataFrame(np.hstack((rmseMean_f.reshape(2, 7).T, rmseStd_f.reshape(2, 7).T)), index=row_labels,
                                columns=col_labels)
    nci_table_f = pd.DataFrame(np.hstack((nciMean_f.reshape(2, 7).T, nciStd_f.reshape(2, 7).T)), index=row_labels,
                               columns=col_labels)
    nll_table_f = pd.DataFrame(np.hstack((nllMean_f.reshape(2, 7).T, nllStd_f.reshape(2, 7).T)), index=row_labels,
                               columns=col_labels)
    rmse_table_s = pd.DataFrame(np.hstack((rmseMean_s.reshape(2, 7).T, rmseStd_s.reshape(2, 7).T)), index=row_labels,
                                columns=col_labels)
    nci_table_s = pd.DataFrame(np.hstack((nciMean_s.reshape(2, 7).T, nciStd_s.reshape(2, 7).T)), index=row_labels,
                               columns=col_labels)
    nll_table_s = pd.DataFrame(np.hstack((nllMean_s.reshape(2, 7).T, nllStd_s.reshape(2, 7).T)), index=row_labels,
                               columns=col_labels)
    # print tables
    print('Filter RMSE')
    print(rmse_table_f)
    print('Filter NCI')
    print(nci_table_f)
    print('Filter NLL')
    print(nll_table_f)
    print('Smoother RMSE')
    print(rmse_table_s)
    print('Smoother NCI')
    print(nci_table_s)
    print('Smoother NLL')
    print(nll_table_s)
    # return computed metrics for filters and smoothers
    return {'filter_RMSE': rmse_table_f, 'filter_NCI': nci_table_f, 'filter_NLL': nll_table_f,
            'smoother_RMSE': rmse_table_s, 'smoother_NCI': nci_table_s, 'smoother_NLL': nll_table_s}


def hypers_demo(lscale=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 1e1, 3e1]):
    steps, mc = 500, 20
    ssm = UNGM()  # initialize UNGM model
    x, z = ssm.simulate(steps, mc_sims=mc)  # generate some data
    # lscale = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 1e1, 3e1]  # , 1e2, 3e2]
    sigmas_ut = Unscented.unit_sigma_points(ssm.xD, kappa=0.0)
    mean_f, cov_f = np.zeros((ssm.xD, steps, mc, len(lscale))), np.zeros((ssm.xD, ssm.xD, steps, mc, len(lscale)))
    for iel, el in enumerate(lscale):
        # initialize BHKF with current lenghtscale
        f = GPQuadKalman(ssm, usp_dyn=sigmas_ut, usp_meas=sigmas_ut,
                         hyp_dyn={'sig_var': 1.0, 'lengthscale': el * np.ones(ssm.xD, ), 'noise_var': 1e-8},
                         hyp_meas={'sig_var': 1.0, 'lengthscale': el * np.ones(ssm.xD, ), 'noise_var': 1e-8})
        # filtering
        for s in range(mc):
            mean_f[..., s, iel], cov_f[..., s, iel] = f.forward_pass(z[..., s])

    # compute average (over MC sims) RMSE and NCI and NLL
    rmseVsEl = rmse(x, mean_f).mean(axis=1)
    nciVsEl = nci(x, mean_f, cov_f).mean(axis=1)
    nllVsEl = nll(x, mean_f, cov_f).mean(axis=1)
    # plot influence of changing lengthscale on the RMSE and NCI and NLL filter performance
    plt.figure()
    plt.semilogx(lscale, rmseVsEl.squeeze(), color='k', ls='-', lw=2, marker='o', label='RMSE')
    plt.semilogx(lscale, nciVsEl.squeeze(), color='k', ls='--', lw=2, marker='o', label='NCI')
    plt.semilogx(lscale, nllVsEl.squeeze(), color='k', ls='-.', lw=2, marker='o', label='NLL')
    plt.grid(True)
    plt.legend()
    plt.show()
    plot_data = {'el': lscale, 'rmse': rmseVsEl, 'nci': nciVsEl, 'nll': nllVsEl}
    return plot_data


if __name__ == '__main__':
    tables_dict = tables()
    plot_data = hypers_demo()
