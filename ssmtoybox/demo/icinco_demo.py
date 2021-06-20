import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from numpy import newaxis as na
from ssmtoybox.ssinf import ExtendedKalman, CubatureKalman, UnscentedKalman, GaussHermiteKalman, GaussianProcessKalman
from ssmtoybox.ssmod import UNGMTransition, UNGMMeasurement
from ssmtoybox.utils import bootstrap_var, squared_error, neg_log_likelihood, log_cred_ratio, mse_matrix, GaussRV


def evaluate_performance(x, mean_f, cov_f, mean_s, cov_s, bootstrap_variance=True):
    num_dim, num_step, num_sim, num_alg = mean_f.shape

    # simulation-average of time-averaged RMSE
    print('RMSE...')
    rmseData_f = np.sqrt(np.mean(squared_error(x[..., na], mean_f), axis=1))
    rmseData_s = np.sqrt(np.mean(squared_error(x[..., na], mean_s), axis=1))
    rmseMean_f = rmseData_f.mean(axis=1).T
    rmseMean_s = rmseData_s.mean(axis=1).T

    print('NLL and NCI...')
    nllData_f = np.zeros((1, num_step, num_sim, num_alg))
    nciData_f = nllData_f.copy()
    nllData_s = nllData_f.copy()
    nciData_s = nllData_f.copy()
    for k in range(1, num_step):
        for fi in range(num_alg):
            mse_mat_f = mse_matrix(x[:, k, :], mean_f[:, k, :, fi])
            mse_mat_s = mse_matrix(x[:, k, :], mean_f[:, k, :, fi])
            for s in range(num_sim):
                # filter scores
                nllData_f[:, k, s, fi] = neg_log_likelihood(x[:, k, s], mean_f[:, k, s, fi], cov_f[:, :, k, s, fi])
                nciData_f[:, k, s, fi] = log_cred_ratio(x[:, k, s], mean_f[:, k, s, fi], cov_f[:, :, k, s, fi],
                                                        mse_mat_f)

                # smoother scores
                nllData_s[:, k, s, fi] = neg_log_likelihood(x[:, k, s], mean_s[:, k, s, fi], cov_s[:, :, k, s, fi])
                nciData_s[:, k, s, fi] = log_cred_ratio(x[:, k, s], mean_s[:, k, s, fi], cov_s[:, :, k, s, fi],
                                                        mse_mat_s)

    nciData_f, nciData_s = nciData_f.mean(axis=1), nciData_s.mean(axis=1)
    nllData_f, nllData_s = nllData_f.mean(axis=1), nllData_s.mean(axis=1)

    # average scores (over time and MC simulations)
    nciMean_f, nciMean_s = nciData_f.mean(axis=1).T, nciData_s.mean(axis=1).T
    nllMean_f, nllMean_s = nllData_f.mean(axis=1).T, nllData_s.mean(axis=1).T

    if bootstrap_variance:
        print('Bootstrapping variance ...')
        num_bs_samples = 10000
        rmseStd_f, rmseStd_s = np.zeros((num_alg, 1)), np.zeros((num_alg, 1))
        nciStd_f, nciStd_s = rmseStd_f.copy(), rmseStd_f.copy()
        nllStd_f, nllStd_s = rmseStd_f.copy(), rmseStd_f.copy()
        for f in range(num_alg):
            rmseStd_f[f] = 2 * np.sqrt(bootstrap_var(rmseData_f[..., f], num_bs_samples))
            rmseStd_s[f] = 2 * np.sqrt(bootstrap_var(rmseData_s[..., f], num_bs_samples))
            nciStd_f[f] = 2 * np.sqrt(bootstrap_var(nciData_f[..., f], num_bs_samples))
            nciStd_s[f] = 2 * np.sqrt(bootstrap_var(nciData_s[..., f], num_bs_samples))
            nllStd_f[f] = 2 * np.sqrt(bootstrap_var(nllData_f[..., f], num_bs_samples))
            nllStd_s[f] = 2 * np.sqrt(bootstrap_var(nllData_s[..., f], num_bs_samples))

        return rmseMean_f, nciMean_f, nllMean_f, rmseMean_s, nciMean_s, nllMean_s, \
               rmseStd_f, nciStd_f, nllStd_f, rmseStd_s, nciStd_s, nllStd_s
    else:
        return rmseMean_f, nciMean_f, nllMean_f, rmseMean_s, nciMean_s, nllMean_s


def print_table(data, row_labels=None, col_labels=None, latex=False):
    pd.DataFrame(data, index=row_labels, columns=col_labels)
    print(pd)
    if latex:
        pd.to_latex()


def tables():
    steps, mc = 500, 100

    # setup univariate non-stationary growth model
    x0 = GaussRV(1, cov=np.atleast_2d(5.0))
    q = GaussRV(1, cov=np.atleast_2d(10.0))
    dyn = UNGMTransition(x0, q)  # dynamics
    r = GaussRV(1)
    obs = UNGMMeasurement(r, 1)  # observation model
    x = dyn.simulate_discrete(steps, mc_sims=mc)  # generate some data
    z = obs.simulate_measurements(x)

    kern_par_sr = np.array([[1.0, 0.3 * dyn.dim_in]])
    kern_par_ut = np.array([[1.0, 3.0 * dyn.dim_in]])
    kern_par_gh = np.array([[1.0, 0.1 * dyn.dim_in]])

    # initialize filters/smoothers
    algorithms = (
        CubatureKalman(dyn, obs),
        UnscentedKalman(dyn, obs),
        GaussHermiteKalman(dyn, obs),
        GaussHermiteKalman(dyn, obs),
        GaussHermiteKalman(dyn, obs),
        GaussHermiteKalman(dyn, obs),
        GaussHermiteKalman(dyn, obs),
        GaussianProcessKalman(dyn, obs, kern_par_sr, kern_par_sr),
        GaussianProcessKalman(dyn, obs, kern_par_ut, kern_par_ut),
        GaussianProcessKalman(dyn, obs, kern_par_sr, kern_par_sr),
        GaussianProcessKalman(dyn, obs, kern_par_gh, kern_par_gh),
        GaussianProcessKalman(dyn, obs, kern_par_gh, kern_par_gh),
        GaussianProcessKalman(dyn, obs, kern_par_gh, kern_par_gh),
        GaussianProcessKalman(dyn, obs, kern_par_gh, kern_par_gh),
    )
    num_algs = len(algorithms)

    # space for estimates
    mean_f, cov_f = np.zeros((dyn.dim_in, steps, mc, num_algs)), np.zeros((dyn.dim_in, dyn.dim_in, steps, mc, num_algs))
    mean_s, cov_s = np.zeros((dyn.dim_in, steps, mc, num_algs)), np.zeros((dyn.dim_in, dyn.dim_in, steps, mc, num_algs))
    # do filtering/smoothing
    t0 = time.time()  # measure execution time
    print('Running filters/smoothers ...', flush=True)
    for a, alg in enumerate(algorithms):
        for sim in trange(mc, desc='{:25}'.format(alg.__class__.__name__), file=sys.stdout):
            mean_f[..., sim, a], cov_f[..., sim, a] = alg.forward_pass(z[..., sim])
            mean_s[..., sim, a], cov_s[..., sim, a] = alg.backward_pass()
            alg.reset()
    print('Done in {0:.4f} [sec]'.format(time.time() - t0))

    # evaluate perfomance
    scores = evaluate_performance(x, mean_f, cov_f, mean_s, cov_s)
    rmseMean_f, nciMean_f, nllMean_f, rmseMean_s, nciMean_s, nllMean_s = scores[:6]
    rmseStd_f, nciStd_f, nllStd_f, rmseStd_s, nciStd_s, nllStd_s = scores[6:]

    # put data into Pandas DataFrame for fancy printing and latex export
    row_labels = ['SR', 'UT', 'GH-5', 'GH-7', 'GH-10', 'GH-15',
                  'GH-20']  # [alg.__class__.__name__ for alg in algorithms]
    col_labels = ['Classical', 'Bayesian', 'Classical (2std)', 'Bayesian (2std)']
    rmse_table_f = pd.DataFrame(np.hstack((rmseMean_f.reshape(2, 7).T, rmseStd_f.reshape(2, 7).T)),
                                index=row_labels, columns=col_labels)
    nci_table_f = pd.DataFrame(np.hstack((nciMean_f.reshape(2, 7).T, nciStd_f.reshape(2, 7).T)),
                               index=row_labels, columns=col_labels)
    nll_table_f = pd.DataFrame(np.hstack((nllMean_f.reshape(2, 7).T, nllStd_f.reshape(2, 7).T)),
                               index=row_labels, columns=col_labels)
    rmse_table_s = pd.DataFrame(np.hstack((rmseMean_s.reshape(2, 7).T, rmseStd_s.reshape(2, 7).T)),
                                index=row_labels, columns=col_labels)
    nci_table_s = pd.DataFrame(np.hstack((nciMean_s.reshape(2, 7).T, nciStd_s.reshape(2, 7).T)),
                               index=row_labels, columns=col_labels)
    nll_table_s = pd.DataFrame(np.hstack((nllMean_s.reshape(2, 7).T, nllStd_s.reshape(2, 7).T)),
                               index=row_labels, columns=col_labels)
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


def hypers_demo(lscale=None):
    # set default lengthscales if unspecified
    if lscale is None:
        lscale = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 1e1, 3e1]

    steps, mc = 500, 20

    # setup univariate non-stationary growth model
    x0 = GaussRV(1, cov=np.atleast_2d(5.0))
    q = GaussRV(1, cov=np.atleast_2d(10.0))
    dyn = UNGMTransition(x0, q)  # dynamics
    r = GaussRV(1)
    obs = UNGMMeasurement(r, 1)  # observation model
    x = dyn.simulate_discrete(steps, mc_sims=mc)  # generate some data
    z = obs.simulate_measurements(x)

    num_el = len(lscale)
    mean_f, cov_f = np.zeros((dyn.dim_in, steps, mc, num_el)), np.zeros((dyn.dim_in, dyn.dim_in, steps, mc, num_el))
    for iel, el in enumerate(lscale):

        # kernel parameters
        ker_par = np.array([[1.0, el * dyn.dim_in]])

        # initialize BHKF with current lenghtscale
        f = GaussianProcessKalman(dyn, obs, ker_par, ker_par)
        # filtering
        for s in range(mc):
            mean_f[..., s, iel], cov_f[..., s, iel] = f.forward_pass(z[..., s])

    # evaluate RMSE, NCI and NLL
    rmseVsEl = squared_error(x[..., na], mean_f)
    nciVsEl = rmseVsEl.copy()
    nllVsEl = rmseVsEl.copy()
    for k in range(steps):
        for iel in range(num_el):
            mse_mat = mse_matrix(x[:, k, :], mean_f[:, k, :, iel])
            for s in range(mc):
                nciVsEl[:, k, s, iel] = log_cred_ratio(x[:, k, s], mean_f[:, k, s, iel], cov_f[:, :, k, s, iel],
                                                       mse_mat)
                nllVsEl[:, k, s, iel] = neg_log_likelihood(x[:, k, s], mean_f[:, k, s, iel], cov_f[:, :, k, s, iel])

    # average out time and MC simulations
    rmseVsEl = np.sqrt(np.mean(rmseVsEl, axis=1)).mean(axis=1)
    nciVsEl = nciVsEl.mean(axis=(1, 2))
    nllVsEl = nllVsEl.mean(axis=(1, 2))

    # plot influence of changing lengthscale on the RMSE and NCI and NLL filter performance
    plt.figure()
    plt.semilogx(lscale, rmseVsEl.squeeze(), color='k', ls='-', lw=2, marker='o', label='RMSE')
    plt.semilogx(lscale, nciVsEl.squeeze(), color='k', ls='--', lw=2, marker='o', label='NCI')
    plt.semilogx(lscale, nllVsEl.squeeze(), color='k', ls='-.', lw=2, marker='o', label='NLL')
    plt.grid(True)
    plt.legend()
    plt.show()

    return {'el': lscale, 'rmse': rmseVsEl, 'nci': nciVsEl, 'neg_log_likelihood': nllVsEl}


if __name__ == '__main__':
    # tables_dict = tables()
    plot_data = hypers_demo()
