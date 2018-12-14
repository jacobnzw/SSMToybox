import time

import numpy as np
import pandas as pd

from ssmtoybox.demo.icinco_demo import evaluate_performance
from ssmtoybox.ssinf import ExtendedKalman, ExtendedKalmanGPQD
from ssmtoybox.ssmod import UNGMTransition, UNGMMeasurement
from ssmtoybox.mtran import UnscentedTransform
from ssmtoybox.utils import GaussRV

steps, mc = 50, 10  # time steps, mc simulations

# setup univariate non-stationary growth model
x0 = GaussRV(1, cov=np.atleast_2d(5.0))
q = GaussRV(1, cov=np.atleast_2d(10.0))
dyn = UNGMTransition(x0, q)  # dynamics
r = GaussRV(1)
obs = UNGMMeasurement(r, 1)  # observation model

x = dyn.simulate_discrete(steps, mc)
z = obs.simulate_measurements(x)

# use only the central sigma-point
usp_0 = np.zeros((dyn.dim_in, 1))
usp_ut = UnscentedTransform.unit_sigma_points(dyn.dim_in)

# set the RBF kernel hyperparameters
hyp_rbf = np.array([[1.0] + dyn.dim_in*[3.0]])
hyp_rbf_ut = np.array([[8.0] + dyn.dim_in*[0.5]])

# derivative observations only at the central point
der_mask = np.array([0])

# filters/smoothers to test
algorithms = (
    # EKF, GPQ+D w/ affine kernel, GPQ+D w/ RBF kernel (el --> infty)
    ExtendedKalman(dyn, obs),
    # GPQ+D RBF kernel w/ single sigma-point, becomes EKF for el --> infinity
    ExtendedKalmanGPQD(dyn, obs, hyp_rbf, hyp_rbf),
)
num_alg = len(algorithms)

# space for estimates
mean_f, cov_f = np.zeros((dyn.dim_in, steps, mc, num_alg)), np.zeros((dyn.dim_in, dyn.dim_in, steps, mc, num_alg))
mean_s, cov_s = np.zeros((dyn.dim_in, steps, mc, num_alg)), np.zeros((dyn.dim_in, dyn.dim_in, steps, mc, num_alg))

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

# rmseMean_f, rmseMean_s = rmseMean_f.squeeze(), rmseMean_s.squeeze()
# nciMean_f, nciMean_s = nciMean_f.squeeze(), nciMean_s.squeeze()
# nllMean_f, nllMean_s = nllMean_f.squeeze(), nllMean_s.squeeze()

# put data into Pandas DataFrame for fancy printing and latex export
row_labels = ['EKF', 'EKF-GPQD']  # ['EKF', 'GPQD-RBF', 'GPQD-AFFINE', 'UKF', 'GPQD-UT-RBF']
col_labels = ['RMSE', '2STD', 'NCI', '2STD', 'NLL', '2STD']
pd.set_option('precision', 4, 'max_columns', 6)
table_f = pd.DataFrame(np.hstack((rmseMean_f, rmseStd_f, nciMean_f, nciStd_f, nllMean_f, nllStd_f)),
                       index=row_labels, columns=col_labels)
table_s = pd.DataFrame(np.hstack((rmseMean_s, rmseStd_s, nciMean_s, nciStd_s, nllMean_s, nllStd_s)),
                       index=row_labels, columns=col_labels)
# print tables
print('Filter metrics')
print(table_f)
print('Smoother metrics')
print(table_s)
