import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import newaxis as na
from scipy.stats import multivariate_normal
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import cholesky

from ssmtoybox.ssinf import ExtendedKalman, CubatureKalman, UnscentedKalman, GaussHermiteKalman
from ssmtoybox.ssinf import GaussianProcessKalman, BayesSardKalman, BayesSardTransform
from ssmtoybox.ssmod import UNGMGaussSSM
from ssmtoybox.utils import bootstrap_var, squared_error, neg_log_likelihood, log_cred_ratio, mse_matrix

alg_label_dict = {
    'GaussianProcessKalman': 'GPQKF',
    'BayesSardKalman': 'BSQKF',
    'ExtendedKalman': 'EKF',
    'CubatureKalman': 'CKF',
    'UnscentedKalman': 'UKF',
    'GaussHermiteKalman': 'GHKF',
}


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
    ssm = UNGMGaussSSM()  # initialize UNGM model
    x, z = ssm.simulate(steps, mc_sims=mc)  # generate some data

    par_sr = np.array([[5.0, 0.3]])
    par_ut = np.array([[5.0, 0.5]])
    par_gh = np.array([[4.0, 0.1]])
    mulind_ut = np.array([[0, 1, 2]])
    mulind_gh = lambda degree: np.atleast_2d(np.arange(degree))

    # initialize filters/smoothers
    algorithms = (
        # ExtendedKalman(ssm),
        # CubatureKalman(ssm),
        UnscentedKalman(ssm, alpha=1.0, beta=0.0),
        GaussHermiteKalman(ssm, deg=5),
        GaussHermiteKalman(ssm, deg=7),
        # GaussHermiteKalman(ssm, deg=10),
        # GaussHermiteKalman(ssm, deg=15),
        # GaussHermiteKalman(ssm, deg=20),
        # BayesSardKalman(ssm, par_sr, par_sr, points='sr'),
        BayesSardKalman(ssm, par_ut, par_ut, mulind_ut, mulind_ut, points='ut', point_hyp={'alpha': 1.0}),
        BayesSardKalman(ssm, par_sr, par_sr, mulind_gh(5), mulind_gh(5), points='gh', point_hyp={'degree': 5}),
        BayesSardKalman(ssm, par_gh, par_gh, mulind_gh(7), mulind_gh(7), points='gh', point_hyp={'degree': 7}),
        # BayesSardKalman(ssm, par_gh, par_gh, points='gh', point_hyp={'degree': 10}),
        # BayesSardKalman(ssm, par_gh, par_gh, points='gh', point_hyp={'degree': 15}),
        # BayesSardKalman(ssm, par_gh, par_gh, points='gh', point_hyp={'degree': 20}),
        # GaussianProcessKalman(ssm, par_sr, par_sr, kernel='rbf', points='sr'),
        GaussianProcessKalman(ssm, par_ut, par_ut, kernel='rbf', points='ut', point_hyp={'alpha': 1.0}),
        GaussianProcessKalman(ssm, par_sr, par_sr, kernel='rbf', points='gh', point_hyp={'degree': 5}),
        GaussianProcessKalman(ssm, par_gh, par_gh, kernel='rbf', points='gh', point_hyp={'degree': 7}),
        # GaussianProcessKalman(ssm, par_gh, par_gh, kernel='rbf', points='gh', point_hyp={'degree': 10}),
        # GaussianProcessKalman(ssm, par_gh, par_gh, kernel='rbf', points='gh', point_hyp={'degree': 15}),
        # GaussianProcessKalman(ssm, par_gh, par_gh, kernel='rbf', points='gh', point_hyp={'degree': 20}),
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
    scores = evaluate_performance(x, mean_f, cov_f, mean_s, cov_s)
    rmseMean_f, nciMean_f, nllMean_f, rmseMean_s, nciMean_s, nllMean_s = scores[:6]
    rmseStd_f, nciStd_f, nllStd_f, rmseStd_s, nciStd_s, nllStd_s = scores[6:]

    # put data into Pandas DataFrame for fancy printing and latex export
    # row_labels = ['SR', 'UT', 'GH-5', 'GH-7', 'GH-10', 'GH-15', 'GH-20']
    row_labels = ['UT', 'GH-5', 'GH-7']
    num_labels = len(row_labels)
    col_labels = ['Classical', 'BSQ', 'GPQ', 'Classical (2std)', 'BSQ (2std)', 'GPQ (2std)']
    rmse_table_f = pd.DataFrame(np.hstack((rmseMean_f.reshape(3, num_labels).T, rmseStd_f.reshape(3, num_labels).T)),
                                index=row_labels, columns=col_labels)
    nci_table_f = pd.DataFrame(np.hstack((nciMean_f.reshape(3, num_labels).T, nciStd_f.reshape(3, num_labels).T)),
                               index=row_labels, columns=col_labels)
    nll_table_f = pd.DataFrame(np.hstack((nllMean_f.reshape(3, num_labels).T, nllStd_f.reshape(3, num_labels).T)),
                               index=row_labels, columns=col_labels)
    rmse_table_s = pd.DataFrame(np.hstack((rmseMean_s.reshape(3, num_labels).T, rmseStd_s.reshape(3, num_labels).T)),
                                index=row_labels, columns=col_labels)
    nci_table_s = pd.DataFrame(np.hstack((nciMean_s.reshape(3, num_labels).T, nciStd_s.reshape(3, num_labels).T)),
                               index=row_labels, columns=col_labels)
    nll_table_s = pd.DataFrame(np.hstack((nllMean_s.reshape(3, num_labels).T, nllStd_s.reshape(3, num_labels).T)),
                               index=row_labels, columns=col_labels)
    # print tables
    pd.set_option('precision', 2, 'display.max_columns', 6)
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


# TODO: plot EMV vs. ell on lower dimensional problem
def lengthscale_filter_demo(lscale):
    steps, mc = 500, 20
    ssm = UNGMGaussSSM()  # initialize UNGM model
    x, z = ssm.simulate(steps, mc_sims=mc)  # generate some data
    num_el = len(lscale)
    # lscale = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 1e1, 3e1]  # , 1e2, 3e2]
    mean_f, cov_f = np.zeros((ssm.xD, steps, mc, num_el)), np.zeros((ssm.xD, ssm.xD, steps, mc, num_el))
    for iel, el in enumerate(lscale):

        # kernel parameters
        ker_par = np.array([[1.0, el * ssm.xD]])

        # initialize BHKF with current lenghtscale
        f = GaussianProcessKalman(ssm, ker_par, ker_par, kernel='rbf', points='ut')
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

    plot_data = {'el': lscale, 'rmse': rmseVsEl, 'nci': nciVsEl, 'neg_log_likelihood': nllVsEl}
    return plot_data


def lengthscale_demo(lscale, two_dim=False):
    alpha_ut = np.array([[0, 1, 2]])
    tf = BayesSardTransform(1, 1, np.array([[1, 1]]), alpha_ut, point_str='ut')

    emv = np.zeros((len(lscale)))
    for i, ell in enumerate(lscale):
        par = np.array([[1.0, ell]])
        emv[i] = tf.model.exp_model_variance(par, alpha_ut)

    plt.figure()
    plt.semilogx(lscale, emv)
    plt.xlabel('$\ell$')
    plt.ylabel('EMV')
    plt.show()

    # 2D case
    if two_dim:
        alpha_ut = np.hstack((np.zeros((2, 1)), np.eye(2), 2*np.eye(2))).astype(np.int)
        tf = BayesSardTransform(2, 1, np.array([[1, 1, 1]]), alpha_ut, point_str='ut')
        emv = np.zeros((len(lscale), len(lscale)))
        for i, ell_0 in enumerate(lscale):
            for j, ell_1 in enumerate(lscale):
                par = np.array([[1.0, ell_0, ell_1]])
                emv[i, j] = tf.model.exp_model_variance(par, alpha_ut)

        fig = plt.figure()
        from mpl_toolkits.mplot3d.axes3d import Axes3D
        ax = Axes3D(fig)
        X, Y = np.meshgrid(np.log10(lscale), np.log10(lscale))
        ax.plot_surface(X, Y, emv)

        ax.set_xlabel('$\log_{10}(\ell_1)$')
        ax.set_ylabel('$\log_{10}(\ell_2)$')
        ax.set_zlabel('EMV')
        plt.show()


if __name__ == '__main__':
    # TODO: use argsparse to create nice command line interface
    tables_dict = tables()
    # save tables in LaTeX format
    pd.set_option('precision', 2)
    with open('ungm_rmse.tex', 'w') as file:
        tables_dict['filter_RMSE'].to_latex(file, float_format=lambda s: '{:.3f}'.format(s))
    with open('ungm_inc.tex', 'w') as file:
        tables_dict['filter_NCI'].to_latex(file, float_format=lambda s: '{:.3f}'.format(s))
    with open('ungm_nll.tex', 'w') as file:
        tables_dict['filter_NLL'].to_latex(file, float_format=lambda s: '{:.3f}'.format(s))

    # lscales = np.logspace(-3, 3, 100)
    # plot_data = lengthscale_filter_demo(lscales)

    # lengthscale_demo(lscales)
