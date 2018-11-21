from unittest import TestCase

import numpy as np
import numpy.linalg as la

from ssmtoybox.ssinf import GPQMKalman, BayesSardKalman
from ssmtoybox.ssmod import UNGMTransition, UNGMMeasurement, Pendulum2DTransition, Pendulum2DMeasurement


class BSQKalmanTest(TestCase):
    def test_init(self):
        mod_dyn = UNGMTransition()
        mod_meas = UNGMMeasurement()
        kpar = np.array([[1, 1]], dtype=np.float)
        alpha = np.array([[0, 1, 2]])
        alg = BayesSardKalman(mod_dyn, mod_meas, kpar, kpar)

    def test_filtering_ungm(self):
        mod_dyn = UNGMTransition()
        mod_meas = UNGMMeasurement()
        kpar = np.array([[1, 1]], dtype=np.float)
        alpha = np.array([[0, 1, 2]])
        alg = BayesSardKalman(mod_dyn, mod_meas, kpar, kpar)
        x = mod_dyn.simulate_discrete(100)
        y = mod_meas.simulate_measurements(x)
        alg.forward_pass(y[..., 0])

    def test_filtering_pendulum(self):
        mod_dyn = Pendulum2DTransition()
        mod_meas = Pendulum2DMeasurement()
        kpar = np.array([[1, 1, 1]], dtype=np.float)
        alpha = np.array([[0, 1, 0, 2, 0],
                          [0, 0, 1, 0, 2]])
        alg = BayesSardKalman(mod_dyn, mod_meas, kpar, kpar)
        x = mod_dyn.simulate_discrete(100)
        y = mod_meas.simulate_measurements(x)
        alg.forward_pass(y[..., 0])


class GPQMarginalizedTest(TestCase):
    def test_init(self):
        ssm = UNGMGaussSSM()
        alg = GPQMKalman(ssm, mod_meas, 'rbf', 'sr')

    def test_time_update(self):
        ssm = UNGMGaussSSM()
        alg = GPQMKalman(ssm, mod_meas, 'rbf', 'sr')
        alg._time_update(1)
        par_dyn, par_obs = np.array([1, 1]), np.array([1, 1])
        alg._time_update(1, par_dyn, par_obs)

    def test_laplace_approx(self):
        ssm = UNGMGaussSSM()
        alg = GPQMKalman(ssm, mod_meas, 'rbf', 'sr')
        # Random measurement
        y = np.sqrt(10)*np.random.randn(1)
        alg._param_posterior_moments(y, 10)
        la.cholesky(alg.param_cov)

    def test_measurement_update(self):
        ssm = UNGMGaussSSM()
        ssm_state, ssm_observations = ssm.simulate(5)
        alg = GPQMKalman(ssm, mod_meas, 'rbf', 'sr')
        alg._measurement_update(ssm_observations[:, 0, 0], 1)

    def test_filtering_ungm(self):
        ssm = UNGMGaussSSM()
        ssm_state, ssm_observations = ssm.simulate(100)
        alg = GPQMKalman(ssm, mod_meas, 'rbf', 'sr')
        alg.forward_pass(ssm_observations[..., 0])

    def test_filtering_pendulum(self):
        ssm = PendulumGaussSSM()
        ssm_state, ssm_observations = ssm.simulate(100)
        alg = GPQMKalman(ssm, mod_meas, 'rbf', 'sr')
        alg.forward_pass(ssm_observations[..., 0])