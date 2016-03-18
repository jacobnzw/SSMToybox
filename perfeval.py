import numpy as np
import pandas as pd
from numpy import newaxis as na
from scipy.stats import multivariate_normal
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import cholesky


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
    MSE = (dx ** 2).mean(axis=1)
    return np.sqrt(MSE)[:, 1:, ...]


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


def print_table(data, row_labels=None, col_labels=None, latex=False):
    pd.DataFrame(data, index=row_labels, columns=col_labels)
    print pd
    if latex:
        pd.to_latex()


# def main():
from inference import ExtendedKalman, CubatureKalman, UnscentedKalman, GaussHermiteKalman, GPQuadKalman
from transforms import SphericalRadial, Unscented, GaussHermite, GPQuad
from models.ungm import UNGM
import time
import scipy.io as sio

# mat = sio.loadmat('d:\\Dropbox\\sources\matlab\\test_lab\\BHKF_testing\\ungm_data_500N_100MC.mat')

steps, mc = 500, 100
ssm = UNGM()  # initialize UNGM model
x, z = ssm.simulate(steps, mc_sims=mc)  # generate some data
# x, z = mat['x'][:, :steps, :], mat['z'][:, :steps, :]
# initialize filters/smoothers
algorithms = (
    # ExtendedKalman(ssm),
    CubatureKalman(ssm),
    UnscentedKalman(ssm, kappa=0.0),
    GaussHermiteKalman(ssm, deg=5),
    GaussHermiteKalman(ssm, deg=7),
    GaussHermiteKalman(ssm, deg=10),
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
)
num_algs = len(algorithms)
# space for estimates
mean_f, cov_f = np.zeros((ssm.xD, steps, mc, num_algs)), np.zeros((ssm.xD, ssm.xD, steps, mc, num_algs))
mean_s, cov_s = np.zeros((ssm.xD, steps, mc, num_algs)), np.zeros((ssm.xD, ssm.xD, steps, mc, num_algs))

# do filtering/smoothing
t0 = time.time()  # measure execution time
print 'Running filters/smoothers ...'
for a, alg in enumerate(algorithms):
    print '{}'.format(alg.__class__.__name__)  # print filter/smoother name
    for sim in range(mc):
        mean_f[..., sim, a], cov_f[..., sim, a] = alg.forward_pass(z[..., sim])
        mean_s[..., sim, a], cov_s[..., sim, a] = alg.backward_pass()
        alg.reset()
print 'Done in {0:.4f} [sec]'.format(time.time() - t0)

# evaluate perfomance
rmseData_f = rmse(x, mean_f).mean(axis=1).T  # averaged filter RMSE over time steps
rmseData_s = rmse(x, mean_s).mean(axis=1).T  # averaged smoother RMSE over time steps
nciData_f = nci(x, mean_f, cov_f).mean(axis=1).T
nciData_s = nci(x, mean_s, cov_s).mean(axis=1).T
data_f = np.hstack((rmseData_f, nciData_f))
data_s = np.hstack((rmseData_s, nciData_s))
# put data into Pandas DataFrame for fancy printing and latex export
row_labels = ['SR', 'UT', 'GH-5', 'GH-7', 'GH-10']  # [alg.__class__.__name__ for alg in algorithms]
col_labels = ['Classical', 'Bayesian']
rmse_table_f = pd.DataFrame(rmseData_f.reshape(2, 5).T, index=row_labels, columns=col_labels)
nci_table_f = pd.DataFrame(nciData_f.reshape(2, 5).T, index=row_labels, columns=col_labels)
rmse_table_s = pd.DataFrame(rmseData_s.reshape(2, 5).T, index=row_labels, columns=col_labels)
nci_table_s = pd.DataFrame(nciData_s.reshape(2, 5).T, index=row_labels, columns=col_labels)
print 'Filter RMSE';
print rmse_table_f
print 'Filter NCI';
print nci_table_f
print 'Smoother RMSE';
print rmse_table_s
print 'Smoother NCI';
print nci_table_s

# table.to_latex()


# if __name__ == '__main__':
#     main()
