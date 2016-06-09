# Evaluate performance of the following filters/smoothers
#   - EKF
#   - GPQKF w/ RBF kernel, one unit sigma-point (zero) and derivative observation
#   - GPQKF w/ AFFINE kernel, one unit sigma-point (zero) and derivative observation
#       * By using affine kernel we hope to recover an algorithm similar to EKF
#       * Difference to EKF is that this algorithm utilizes integral variance
#       * It could be construed as "Bayesian Quadrature EKF" (GP quadrature EKF)
#   - GPQKF w/ given kernel, UT unit sigma-points w/ derivative observation at the middle point ONLY
#       Kernels could be:
#       * RBF:
#       * AFFINE: This variant should be closest to the EKF
#       * HERMITE:

import time

import numpy as np
import pandas as pd

from icinco_demo import rmse, nci, nll, bootstrap_var
from inference import ExtendedKalman, ExtendedKalmanGPQD
from models.ungm import UNGM
from transforms import Unscented

steps, mc = 500, 100  # time steps, mc simulations
# initialize SSM and generate some data
ssm = UNGM()
x, z = ssm.simulate(steps, mc)
# use only the central sigma-point
usp_0 = np.zeros((ssm.xD, 1))
usp_ut = Unscented.unit_sigma_points(ssm.xD)
# set the RBF kernel hyperparameters
hyp_rbf = {'sig_var': 1.0, 'lengthscale': 3.0 * np.ones(ssm.xD, ), 'noise_var': 1e-8}
hyp_rbf_ut = {'sig_var': 8.0, 'lengthscale': 0.5 * np.ones((1,)), 'noise_var': 1e-16}
hyp_affine = {'bias': 1.0, 'variance': 1.0 * np.ones(ssm.xD, ), 'noise_var': 1e-16}
# derivative observations only at the central point
der_mask = np.array([0])
# filters/smoothers to test
algorithms = (
    # EKF, GPQ+D w/ affine kernel, GPQ+D w/ RBF kernel (el --> infty)
    ExtendedKalman(ssm),
    # GPQ+D RBF kernel w/ single sigma-point, becomes EKF for el --> infinity
    ExtendedKalmanGPQD(ssm, el=1.0),
    # GPQ+D affine kernel w/ single sigma-point, x = m + L*xi
    # GPQuadDerAffineKalman(ssm, usp_0, usp_0, hyp_affine, hyp_affine, which_der=der_mask),
    # GPQ+D RBF kernel w/ single sigma-point, x = m + L*xi
    # GPQuadDerRBFKalman(ssm, usp_0, usp_0, hyp_rbf, hyp_rbf, which_der=der_mask),
    # UKF
    # UnscentedKalman(ssm, kappa=0.0),
    # GPQ-UT w/ UT sigma-points, should be same as UKF
    # GPQ-RBF w/ UT sigma-points
    # GPQKalman(ssm, usp_ut, usp_ut, hyp_rbf_ut, hyp_rbf_ut),
    # GPQ+D RBF kernel w/ UT sigma-points (derivative at the central point only)
    # GPQuadDerRBFKalman(ssm, usp_ut, usp_ut, hyp_rbf_ut, hyp_rbf_ut, which_der=der_mask),
    # GPQ+D Hermite kernel w/ UT sigma-points (derivative at the central point only)
    # GPQ+D affine kernel w/ UT sigma-points (could be crap)
)
num_alg = len(algorithms)

# space for estimates
mean_f, cov_f = np.zeros((ssm.xD, steps, mc, num_alg)), np.zeros((ssm.xD, ssm.xD, steps, mc, num_alg))
mean_s, cov_s = np.zeros((ssm.xD, steps, mc, num_alg)), np.zeros((ssm.xD, ssm.xD, steps, mc, num_alg))
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
rmseData_f, rmseData_s = rmse(x, mean_f), rmse(x, mean_s)  # averaged RMSE over time steps
nciData_f, nciData_s = nci(x, mean_f, cov_f), nci(x, mean_s, cov_s)  # averaged NCI over time steps
nllData_f, nllData_s = nll(x, mean_f, cov_f), nll(x, mean_s, cov_s)  # averaged NLL over time steps
# average scores (over MC simulations)
rmseMean_f, rmseMean_s = rmseData_f.mean(axis=1).T, rmseData_s.mean(axis=1).T
nciMean_f, nciMean_s = nciData_f.mean(axis=1).T, nciData_s.mean(axis=1).T
nllMean_f, nllMean_s = nllData_f.mean(axis=1).T, nllData_s.mean(axis=1).T
# +/- 2 standard deviations of the scores (using bootstrapping)
rmseStd_f, rmseStd_s = np.zeros((num_alg, 1)), np.zeros((num_alg, 1))
nciStd_f, nciStd_s = rmseStd_f.copy(), rmseStd_f.copy()
nllStd_f, nllStd_s = rmseStd_f.copy(), rmseStd_f.copy()
for f in range(num_alg):
    rmseStd_f[f, :] = bootstrap_var(rmseData_f[..., f], samples=1e4)
    rmseStd_s[f, :] = bootstrap_var(rmseData_s[..., f], samples=1e4)
    nciStd_f[f, :] = bootstrap_var(nciData_f[..., f], samples=1e4)
    nciStd_s[f, :] = bootstrap_var(nciData_s[..., f], samples=1e4)
    nllStd_f[f, :] = bootstrap_var(nllData_f[..., f], samples=1e4)
    nllStd_s[f, :] = bootstrap_var(nllData_s[..., f], samples=1e4)

# put data into Pandas DataFrame for fancy printing and latex export
row_labels = None  # ['EKF', 'GPQD-RBF', 'GPQD-AFFINE', 'UKF', 'GPQD-UT-RBF']
col_labels = ['RMSE', '2STD', 'NCI', '2STD', 'NLL', '2STD']
pd.set_option('precision', 4)
table_f = pd.DataFrame(np.hstack((rmseMean_f, rmseStd_f, nciMean_f, nciStd_f, nllMean_f, nllStd_f)),
                       index=row_labels, columns=col_labels)
table_s = pd.DataFrame(np.hstack((rmseMean_s, rmseStd_s, nciMean_s, nciStd_s, nllMean_s, nllStd_s)),
                       index=row_labels, columns=col_labels)
# print tables
print 'Filter metrics'
print table_f
print 'Smoother metrics'
print table_s
